import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

torch.manual_seed(1)

# ===== HYPERPARAMETERS =====

BATCH_SIZE = 32 # how many independent sequences will we process in parallel?
BLOCK_SIZE = 8 # what is the max number of tokens we can use as input for a prediction?
               # NOTE: in a block size of n tokens, we will iterate our prediction on n-1 tokens
TRAINING_DATA_PROPORTION = 0.9
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # run on gpu if available
EVAL_ITERS = 200
N_EMBD = 384 # embedding size/dimensions
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2


# ===== CORPUS RETRIEVAL =====
# get the text corpus
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get the unique characters in the text (vocabulary set)
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)


# ===== TOKENISATION =====
# NOTE: Opportunity to enhance. For now, we will just do a new token per char.

# map each character to an integer (token identifier)
char_to_int = { ch:i for i,ch in enumerate(chars) } 
int_to_char = { i:ch for i,ch in enumerate (chars) }

def encode(s: str) -> list[int]:
    """ takes a string and maps each character to an integer"""
    return [char_to_int[c] for c in s]

def decode(l: list[int]) -> str:
    """ takes a list of int and outputs a str"""
    return "".join(int_to_char[i] for i in l)


# split the encoded corpus into train and test data
encoded_corpus = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAINING_DATA_PROPORTION * len(encoded_corpus))
train_data = encoded_corpus[:n]
test_data = encoded_corpus[n:]


def get_batch(split: Literal["train", "test"]):
    """ get numerous sequences of tokens (x) and the corresponding n+1 target tokens (y) """
    # generate a batch of data of inputs x and targets y
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, )) # generates BATCH_SIZE random starting positions
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # get the block of tokens to predict from, for each starting point (matrix)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]) # shift by 1, since n tokens should predict the n+1 token (matrix)
    x,y = x.to(DEVICE), y.to(DEVICE) # move the tensors onto our computation device
    return x, y


@torch.no_grad() # when running this fn, don't calculate gradients
def estimate_loss():
    """ 
    take a number of random batches and avg their losses

    returns: out = { "train": 1.52, "test": 1.68 }
    """
    out = {}    
    model.eval() # put the model in eval mode (some NN's behave differently)
    
    for split in ["train", "test"]:
        losses = torch.zeros(EVAL_ITERS) # [0, 0, 0, 0, 0, ..., 0]
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item() # save the loss number into the array

        # after calculating the loss on EVAL_ITERS predictions, take the mean
        out[split] = losses.mean() 

    model.train() # set the model back to train mode
    return out


# ===== TRANSFORMER  =====
# NOTE: Opportunity to enhance. For now, we will use bigram language model.


class BigramLanguageModel(nn.Module):
    """ A simple language model that predicts the next token from the current token. """

    def __init__(self):
        super().__init__()

        # build 3 lookup tables and fill with random numbers

        # each token value (row) is mapped to an embedding (columns)
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)

        # each token position (row) is mapped to an embedding (columns)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD) # needed for attention

        # initialises a weightght for each embedding cell (columns) across each vocab token (row) + bias
        # it can then be used as a fn that runs logits =  x @ weightght.T + bias
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE) # the final layer (head) of the lm (language model) that produces the predictions

    def forward(self, tok_sequence: torch.Tensor, target_tokens: torch.Tensor=None) -> torch.Tensor:
        """ Generate logits for the next-token predictions. """

        B, T = tok_sequence.shape

        tok_emb = self.token_embedding_table(tok_sequence) # outputs a 3D tensor (B, T, C = N_EMBD)
        pos = torch.arange(T, device=DEVICE) # i.e. torch.arange(3) = tensor([0, 1, 2])
        pos_emb = self.position_embedding_table(pos) # (T, C = N_EMBD)
        x = tok_emb + pos_emb # (B, T, C = N_EMBD) - pos_emb gets repeated B times before adding

        # runs on each of the B x T positions, and produces a logit (score) for each potential output token
        logits = self.lm_head(x) # (B, T, C = VOCAB_SIZE)

        if target_tokens is None:
            loss = None
        else:
            # need to reshape because of the way pytorch expects cross_entropy() inputs
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            target_tokens = target_tokens.view(B*T)

            # Negative Log Likelihood (NLL) loss
            loss = F.cross_entropy(logits, target_tokens)

        return logits, loss

    def generate(self, tok_sequence, max_new_toks):
        """
        Given some tokens, repeatedly predict the next token, add it to the sequence, and repeat.
        This is inference/generation, not training. The weightghts are already trained.
        """
        # tok_sequence is (B, T) array of indices in the current context
        for _ in range(max_new_toks): # (1) the _ (2) the cat _ (3) the cat sat _ (4) the cat sat on _ ...

            # calling self runs self.__call__() which does some processing, then calls forward()
            logits, _ = self(tok_sequence) 

            # focus only on the last time step (we only care about predicting the next token)
            # NOTE: for a bigram model, it is silly that we pass in the full history when it only needs the nth token
            #       however, it will generalise better to other models
            logits = logits[:, -1, :] # becomes (B,C)

            # apply softmax to convert scores into probabilities that sum to 1
            probs = F.softmax(logits, dim=-1) # (B, C)

            # randomly sampling from the distribution (not just taking highest probability token)
            tok_sequence_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append newly generated token to the end of the existing sequence
            tok_sequence = torch.cat((tok_sequence, tok_sequence_next), dim=1) # (B, T+1)

        return tok_sequence
            

model = BigramLanguageModel()
m = model.to(DEVICE)

# ===== TRAINING  =====

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE) # params are the embedding table weightghts (vocab size, vocab size)

ON = False

if ON:
    for iter in range(MAX_ITERS):

        # occasionally evaluate the loss on train and test sets
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

        # sample a batch of data
        xb, yb = get_batch("train")

        # evaluate the loss
        logits, loss = model(xb, yb)

        # erases the gradients from the previous training step
        optimizer.zero_grad(set_to_none=True)

        # calculating this steps gradients (how much should each weightght move to reduce the loss)
        loss.backward()

        # actually change all of the weightghts
        optimizer.step()

    # context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
    # print(decode(m.generate(context, max_new_toks=500)[0].tolist()))


# # ATTENTION
# torch.manual_seed(1337)
# B,T,C = 4,8,2
# x = torch.randn(B,T,C)
# x.shape


# # VERY INEFFICIENT
# # take mean of channels from 1 to t
# xbow = torch.zeros((B,T,C))
# for b in range(B):
#     for t in range(T):
#         xprev = x[b, :t+1] # (t,C)
#         xbow[b,t] = torch.mean(xprev, 0)

# #HENCE WHY WE USE MATRIX MULT
# torch.manual_seed(42)
# a = torch.ones(3,3)
# b = torch.randint(0,10,(3,2)).float()
# c = a @ b

torch.manual_seed(1337)
B, T, C = 4,8,32
x = torch.randn(B,T,C)

tril = torch.tril(torch.ones(T,T))
weight = torch.zeros((T,T))
weight = weight.masked_fill(tril == 0, float('-inf'))

# gives a triangle matrix of evenly dirstributed weights up to the final token before prediction
weight = F.softmax(weight, dim=-1) 

# For each token, and for each channel column, it takes a weighted sum down that column, and the weights are zero for any time step after the current one.
out = weight @ x # embeddings for each token where the embedding weights are adjusted based on the previous tokens

print(out.shape)

# SOFTMAX(z) = (e^z)/sum(e^z)

# for attention, the dot product of the query and the key will become weight