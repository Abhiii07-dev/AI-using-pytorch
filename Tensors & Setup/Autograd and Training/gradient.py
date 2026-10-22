"""
Gradient Descent = improve the number by using the advice of grad
"""

import torch

x = torch.tensor(2.0)

#learnable number
w = torch.tensor(3.0,requires_grad=True)

#Correct tensor

y_true = torch.tensor(10.0)

#Leanrning Rate

lr = 0.1

y_pred = w*x
loss = (y_pred - y_true)**2

loss.backward()

#.item()  extracts the actual number

print("Before update:")
print(f"w:{w.item()}")
print(f"loss:{loss.item()}")
print(f"gradient:{w.grad.item()}")

#Update weight


with torch.no_grad():              #DONOT TRACK THIS  
    w -= lr * w.grad

#IMPORTANT:Reset gradient

w.grad.zero_()         # zeroes the old gradient

print("Aftrt Update")
print(f"w:{w.item()}")




