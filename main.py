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
N_EMBD = 384
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

    # put the model in eval mode (some NN's behave differently)
    model.eval()
    for split in ["train", "test"]:
        losses = torch.zeros(EVAL_ITERS) # [0, 0, 0, 0, 0, ..., 0]
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item() # save the loss number into the array

        # after calculating the loss on EVAL_ITERS predictions, take the mean
        out[split] = losses.mean() 

    model.train()
    return out



# get numerous sequences of tokens (xb) and the corresponding n+1 target tokens (yb)
xb, yb = get_batch("train")






# ===== TRANSFORMER  =====
# NOTE: Opportunity to enhance. For now, we will use bigram language model.


class BigramLanguageModel(nn.Module):
    """ A simple language model that predicts the next token from the current token. """

    def __init__(self):
        super().__init__()

        # each token value (row) is mapped to an embedding (columns)
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)

        # each token position (row) is mapped to an embedding (columns)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)

        # each embedding is then given likelihood scores across the vocab to predict the next token
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE)

    def forward(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor=None) -> torch.Tensor:
        """ 
        Generate logits for the next-token predictions. 

        """

        B, T = input_tokens.shape

        tok_emb = self.token_embedding_table(input_tokens) # outputs a 3D tensor (B, T, C = N_EMBD)
        pos_emb = self.position_embedding_table(torch.arrange(T, device=DEVICE)) # (T,C = N_EMBD)
        x = tok_emb + pos_emb # (B, T, C)
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

    def generate(self, token_sequence, max_new_tokens):
        """
        Given some tokens, repeatedly predict the next token, add it to the sequence, and repeat.
        This is inference/generation, not training. The weights are already trained.
        """
        # token_sequence is (B, T) array of indices in the current context
        for _ in range(max_new_tokens): # (1) the _ (2) the cat _ (3) the cat sat _ (4) the cat sat on _ ...

            # calling self runs self.__call__() which does some processing, then calls forward()
            logits, _ = self(token_sequence) 

            # focus only on the last time step (we only care about predicting the next token)
            # NOTE: for a bigram model, it is silly that we pass in the full history when it only needs the nth token
            #       however, it will generalise better to other models
            logits = logits[:, -1, :] # becomes (B,C)

            # apply softmax to convert scores into probabilities that sum to 1
            probs = F.softmax(logits, dim=-1) # (B, C)

            # randomly sampling from the distribution (not just taking highest probability token)
            token_sequence_next = torch.multinomial(probs, num_samples=1) # (B, 1)

            # append newly generated token to the end of the existing sequence
            token_sequence = torch.cat((token_sequence, token_sequence_next), dim=1) # (B, T+1)

        return token_sequence
            

model = BigramLanguageModel()
m = model.to(DEVICE)

# ===== TRAINING  =====

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE) # params are the embedding table weights (vocab size, vocab size)


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

    # calculating this steps gradients (how much should each weight move to reduce the loss)
    loss.backward()

    # actually change all of the weights
    optimizer.step()

# context = torch.zeros((1,1), dtype=torch.long, device=DEVICE)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


# ATTENTION
torch.manual_seed(1337)
B,T,C = 4,8,2
x = torch.randn(B,T,C)
x.shape


# VERY INEFFICIENT
# take mean of channels from 1 to t
xbow = torch.zeros((B,T,C))
for b in range(B):
    for t in range(T):
        xprev = x[b, :t+1] # (t,C)
        xbow[b,t] = torch.mean(xprev, 0)

#HENCE WHY WE USE MATRIX MULT
torch.manual_seed(42)
a = torch.ones(3,3)
b = torch.randint(0,10,(3,2)).float()
c = a @ b