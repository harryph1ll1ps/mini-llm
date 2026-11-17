import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Literal

torch.manual_seed(1)

# ===== HYPERPARAMETERS =====

TRAINING_DATA_PROPORTION = 0.9

BLOCK_SIZE = 8 # what is the max number of tokens we can use as input for a prediction?
               # NOTE: in a block size of n tokens, we will iterate our prediction on n-1 tokens
BATCH_SIZE = 32 # how many independent sequences will we process in parallel?


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
    x = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]) # get the block of tokens to predict from, for each starting point (matrix)
    y = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]) # shift by 1, since n tokens should predict the n+1 token (matrix)
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

    def forward(self, input_tokens: torch.Tensor, target_tokens: torch.Tensor=None) -> torch.Tensor:
        """ 
        Generate logits for the next-token predictions. 

        params:
        both are (B= BATCH_SIZE, T= BLOCK_SIZE) tensor of integers
        - input_tokens
        - target_tokens
             

        output: 

        """
        # gets the rows (i.e. logit scores for each possible vocab) in the lookup table for each input token (input_tokens)
        logits = self.token_embedding_table(input_tokens) # outputs a 3D tensor: (B= BATCH_SIZE, T= BLOCK_SIZE, C= VOCAB_SIZE)

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
            


m = BigramLanguageModel(vocab_size)
logits, loss = m.forward(xb, yb) 

# print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=50)[0].tolist())) # produces random tokens


# ===== TRAINING  =====

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for steps in range(10000):

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())

# print(decode(m.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=50)[0].tolist())) # getting more readable

