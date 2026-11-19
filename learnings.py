import torch, torch.nn as nn

emb = nn.Embedding(65, 4)
key = nn.Linear(4, 4, bias=False)

# Hash table: same ID → same output, always
print(emb(torch.tensor(20)))
print(emb(torch.tensor(20)))           # identical

# Function: change the input vector → output changes
x1 = torch.tensor([1., 0., 0., 0.])
x2 = torch.tensor([1., 0., 0., 0.5])   # e.g. same char, different position
print(key(x1))
print(key(x2))                          # different