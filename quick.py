#%%
from torch import nn 
import torch

#%%
ce_loss = nn.CrossEntropyLoss()
input = torch.Tensor([[.1,.8,.1],[.1,.2,.3]])
target = torch.Tensor([[1],[2]],dtype=torch.long)
loss = ce_loss(input,target)
# %%
ce_loss = nn.CrossEntropyLoss()
input = torch.Tensor([.1,.8,.1])
target = torch.Tensor([1],dtype=torch.long)
loss = ce_loss(input,target)
# %%
target
# %%
import torch
import torch.nn as nn

# size of input (N x C) is = 3 x 5
input = torch.randn(3, 5, requires_grad=True)
# every element in target should have 0 <= value < C
target = torch.tensor([1, 0, 4])

m = nn.LogSoftmax(dim=1)
nll_loss = nn.NLLLoss()
output = nll_loss(m(input), target)
output.backward()

print('input: ', input)
print('target: ', target)
print('output: ', output)
# %%
