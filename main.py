import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

torch.manual_seed(1)

# ===========================
# ===== HYPERPARAMETERS =====
# ===========================

# careful, larger scale needs a GPU so that mac CPU doesnt get cooked

BATCH_SIZE = 32
BLOCK_SIZE = 64
TRAINING_DATA_PROPORTION = 0.9
MAX_ITERS = 3000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3   # small models tolerate a higher LR
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EVAL_ITERS = 50
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.2

# ============================
# ===== CORPUS RETRIEVAL =====
# ============================

# get the text corpus
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get the unique characters in the text (vocabulary set)
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# ===========================
# ===== TOKENISATION ========
# ===========================

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
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, )) # generates B random starting positions with T space available
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # get the T tokens to predict from, for the starting point of each B (B,T)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]) # shift by 1, since n tokens should predict the n+1 token (B,T)
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

# ===========================
# ===== SELF-ATTENTION  =====
# ===========================

class Head(nn.Module):
    " one head of self-attention "

    def __init__(self, head_size):
        super().__init__()

        # the linear operations that will give the weights for each (B,T) across each "question"/"property" (C = head_size)
        self.key = nn.Linear(N_EMBD, head_size, bias=False) # what I contain? - i.e. embd1: [0.5, 0.7, 0.8,..., 0.9] 
        self.query = nn.Linear(N_EMBD, head_size, bias=False) # what I am searching for?
        self.value = nn.Linear(N_EMBD, head_size, bias=False) # what I hand over?

        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))) # makes the lower triangular matrix

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Causal self-attention: each token takes a weighted mix of the value
        vectors of itself and earlier tokens, with weights from query·key scores.

        x: (B, T, C = N_EMBD) embeddings -> returns (B, T, head_size)
        """
        B, T, C = x.shape
        k = self.key(x) # (B, T, C = head_size) 
                        # NOTE: C gets transformed from N_EMBD to head_size
        q = self.query(x) # (B, T, C = head_size)

        # *** DOT PRODUCT OF QUERY AND KEY BECOMES THE WEIGHT ***
        # [1.0, 0.4,-0.9] dot every token's query with every token's key
        # [0.7, 0.3, 1.2] = affinity scores (how much each token in that row cares
        # [0.2, 0.6, 0.2] about each token in that column [same batch]), future still included
        # divide by sqrt(k) to ensure weights are appropriately diffuse (see attention formula)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        # TRIANGLE (tril)
        # [1, 0, 0] 
        # [1, 1, 0]
        # [1, 1, 1] 
        # MASKED
        # [1, -inf, -inf] turn zeros to -inf using tril
        # [0.7, 0.3,-inf] this prevents any future token from being
        # [0.2, 0.6, 0.2] used to predict the next token after softmax
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)

        # [1.0, 0.0, 0.0] run softmax: e^z/sum(e^z)
        # [0.6, 0.4, 0.0] each row sums to 1, -infs become 0
        # [0.3, 0.4, 0.3] measueres how much each token listens to the prior ones
        wei = F.softmax(wei, dim=-1) # (B, T, T)

        # dropout randomly masks some weights so that the neural net learns sublayers that aggregate in final product
        # see the dropout paper (helps w scale and overfitting)
        wei = self.dropout(wei)

        # what each token actually shares when another token attends to it
        v = self.value(x) # (B, T, C = head_size)

        # *** OUTPUT IS THE NEW CONTEXT-AWARE VECTOR (cols) FOR EACH TOKEN (rows) ***
        # [1.0, 0.0, 0.0]   [1, 4]   [1.0, 4.0] the weighted sum of what the token saw
        # [0.6, 0.4, 0.0] @ [3, 0] = [1.8, 2.4] from previous tokens (v) * how much it cares (wei)
        # [0.3, 0.4, 0.3]   [5, 2]   [3.0, 1.8] 
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)

        return out


# ===========================
# ===== TRANSFORMER  ========
# ===========================

# NOTE: Opportunity to enhance. For now, we will use bigram language model.

class MultiHeadAttention(nn.Module):
    """ 
    A head is 1 self-attention unit. 
    It has its own 3 linear layers (K, Q, V) & computes its own attention pattern.
    This uses multiple heads of self-attention in parallel. 
    Each head can learn to look for something different. 
    Results are concatenated for greater info.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()

        # creates multiple Heads, each with their own K,Q,V weights
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """
        Create multiple self-attention Heads where each token takes a weighted mix 
        of the value vectors of itself and earlier tokens, with weights from query·key scores.

        x: (B, T, C) embeddings -> returns (B, T, num_heads * head_size)
        """
        # runs x through each head (B, T, C = head_size) and joins them along the last dimension (C)
        out = torch.cat([h(x) for h in self.heads], dim=-1) # (B, T, C = num_heads * head_size = N_EMBD)


        # projection back into the residual pathway (merges the concat head_size weights into something meaningful)
        out = self.proj(out) # (B, T, C = N_EMBD)

        out = self.dropout(out)

        return out

class FeedForward(nn.Module):
    """ 
    Self attention is where the tokens see how they interact with one another.
    Feed forward is where each of these tokens then get tuned individually.
    This implementation is a simple linear layer followed by a non-linearity. 
    """

    def __init__(self, N_EMBD):
        super().__init__()
        self.net = nn.Sequential( # do a, then b
            
            # 4 is what is used in "Attention is all you need"
            # expand token vector on residual path (B, T, C = 4 * N_EMBD)
            nn.Linear(N_EMBD, 4 * N_EMBD), # output = input_emb @ weights.T + bias 
                                           # weights begin random, then get learned in training (loss.backward())
            nn.ReLU(), # turn negatives into zeros

            # projection layer going back into residual pathway (B, T, C = N_EMBD)
            nn.Linear(4 * N_EMBD, N_EMBD),

            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        """
        for each token vector independently: blend its numbers into new features (Linear),
        then zero out the negative ones (ReLU)
        x: (B, T, C) -> (B, T, C)
        """
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, N_EMBD, N_HEAD):
        super().__init__()

        # inside a head, each token’s vector is compressed to from N_EMBD to head_size
        head_size = N_EMBD // N_HEAD # used to split the embedding dimensions evenly across N_HEAD

        # communication: run self-attention to get the raw vector numbers
        self.sa = MultiHeadAttention(N_HEAD, head_size)

        # computation: ffwd turns the vector numbers into something more useful for predicting the next token
        self.ffwd = FeedForward(N_EMBD)

        # LayerNorm resets each token’s vector to a consistent scale (mean = 0, stdev = 1) 
        # before it enters attention or the feed-forward.
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.ln2 = nn.LayerNorm(N_EMBD)

    def forward(self, x):
        # residual connections: fork off the main path then connect by addition
        x = x + self.sa(self.ln1(x)) # layer norm (normalisation)
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    """ A simple language model that predicts the next token from the current token. """

    def __init__(self):
        super().__init__()

        # each token value (row) is mapped to an embedding (columns)
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBD)

        # each token position (row) is mapped to an embedding (columns)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD) # needed for attention

        # transformer layers
        self.blocks = nn.Sequential(*[Block(N_EMBD, N_HEAD=N_HEAD) for _ in range(N_LAYER)])
        
        # final layer norm
        self.ln_f = nn.LayerNorm(N_EMBD)

        # the final layer (head) of the lm that converts the context-aware token vector into a score for every vocab
        # initialises a weight for each embedding cell (columns) across each possible vocab token (row) + bias
        # it can then be used as a fn that runs: logits =  x @ weight.T + bias (one score per context-aware token)
                                                         # x is (B, T, N), the output of the transformer blocks
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE) # (B, T, VOCAB_SIZE)


    def forward(self, tok_sequence: torch.Tensor, target_tokens: torch.Tensor=None) -> torch.Tensor:
        """ Generate logits for the next-token predictions. """

        B, T = tok_sequence.shape

        # turn each tok in the sequence into a position aware vector
        tok_emb = self.token_embedding_table(tok_sequence) # outputs a 3D tensor (B, T, C = N_EMBD)
        pos = torch.arange(T, device=DEVICE) # i.e. torch.arange(3) = tensor([0, 1, 2])
        pos_emb = self.position_embedding_table(pos) # (T, C = N_EMBD)
        x = tok_emb + pos_emb # (B, T, C = N_EMBD) - pos_emb gets repeated B times before adding

        # run through transformer layers so each token vector becomes context-aware
        x = self.blocks(x) # (B, T, C)

        # runs on each of the (B, T) positions, and produces a logit (score) for each potential output token (C)
        logits = self.lm_head(x) # (B, T, C = VOCAB_SIZE)

        # calculate the loss based on the logits scores compared to the target tokens
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
        # tok_sequence is (B, T) array of indices
        for _ in range(max_new_toks): # (1) the _ (2) the cat _ (3) the cat sat _ (4) the cat sat on _ ...

            # crop the tok_sequence to ensure it does not exceed BLOCK_SIZE (so that emb tbl does not overflow)
            tok_sequence_cond = tok_sequence[:, -BLOCK_SIZE:]

            # calling self runs self.__call__() which does some processing, then calls forward()
            logits, _ = self(tok_sequence_cond) # (B, T, C)

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

# ===========================
# ===== TRAINING  ===========
# ===========================

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE) # params are the embedding table weights (vocab size, vocab size)

ON = True

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

        # calculating this step's gradients (how much should each weight move to reduce the loss)
        loss.backward()

        # actually change all of the weightghts
        optimizer.step()


# Print: starting from a single token (ID 0) and generate 100 new tokens
context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
generated = m.generate(context, max_new_toks=500)
print(decode(generated[0].tolist()))
