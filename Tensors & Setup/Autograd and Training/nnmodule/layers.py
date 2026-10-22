import torch
import torch.nn as nn

layer = nn.Linear(3,2)

x = torch.tensor([[1.0,2.0,3.0]])

y = layer(x)
print(y)


"""
layer = number transformer
input features = how many numbers came in
output featues = how many numbers returned
weight = decides importnace
biased = how much tuning to match the shapes
"""