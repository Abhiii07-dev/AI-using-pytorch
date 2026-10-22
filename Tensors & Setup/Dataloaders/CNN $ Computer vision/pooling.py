"""
nn.maxPool2d = will ananlyse the most important signal from the batch and discard else
"""


import torch
import torch.nn as nn

x = torch.randn(1,1,28,28)

pool = nn.MaxPool2d(kernel_size = 2)

y = pool(x)

print("before pooling",x.shape)
print("after pooling",y.shape)
