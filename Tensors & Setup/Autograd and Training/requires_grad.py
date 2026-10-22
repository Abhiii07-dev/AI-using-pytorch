import torch

#requires_grad = True/False  = "to track the given data and if in future update is needed it is done automatically"

w = torch.tensor(2.0 , requires_grad= True)
x = torch.tensor(7.0)
p = w*x
print(p)



"""
computation Graph:
it is a hidden map of pytirch that shows how numbers where created
for this to activate , requires_grad = True is required
"""
