import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

torch.manual_seed(1)

# ===== HYPERPARAMETERS =====

TRAINING_DATA_PROPORTION = 0.9

BLOCK_SIZE = 8 # what is the max number of tokens we can use as input for a prediction?
               # NOTE: in a block size of n tokens, we will iterate our prediction on n-1 tokens
BATCH_SIZE = 4 # how many independent sequences will we process in parallel?


# ===== CORPUS RETRIEVAL =====
# get the text corpus
with open('corpus.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# get the unique characters in the text (vocabulary set)
chars = sorted(list(set(text)))
vocab_size = len(chars)


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


# encode the entire corpus and split into train & test data
encoded_corpus = torch.tensor(encode(text), dtype=torch.long)
n = int(TRAINING_DATA_PROPORTION * len(encoded_corpus))
train_data = encoded_corpus[:n]
test_data = encoded_corpus[n:]


def get_batch(split: Literal["train", "test"]):
    # generate a batch of data of inputs x and targets y
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE, )) # generates BATCH_SIZE random starting positions
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # get the block of tokens to predict from, for each starting point
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]) # shift by 1, since n tokens should predict the n+1 token
    return x, y


# get numerous sequences of tokens (xb) and the corresponding n+1 target tokens (yb)
xb, yb = get_batch("train")


# ===== TRANSFORMER  =====
# NOTE: Opportunity to enhance. For now, we will use bigram language model.


class BigramLanguageModel(nn.Module):
    """ A simple language model that predicts the next token from the current token. """

    def __init__(self, vocab_size: int):
        super().__init__()

        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """ 
        Generate logits for the next-token predictions. 

        params:
        both are (B= BATCH_SIZE, T= BLOCK_SIZE) tensor of integers
        - idx
        - targets
             

        output: 

        """

        # gets the rows (i.e. logit scores for each possible vocab) in the lookup table for each input token (idx)
        logits = self.token_embedding_table(idx) # 3D tensor: (B= BATCH_SIZE, T= BLOCK_SIZE, C= VOCAB_SIZE)


        # need to reshape because of the way pytorch expects cross_entropy() inputs
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)

        # Negative Log Likelihood (NLL) loss
        loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Given some tokens, repeatedly predict the next token, add it to the sequence, and repeat.
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B,C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
            


m = BigramLanguageModel(vocab_size)
logits, loss = m.forward(xb, yb) 
