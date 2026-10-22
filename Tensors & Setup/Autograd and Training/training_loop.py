"""
1. make prediction
2. measure loss
3. use backward
4. fix the number suing gradient 
5. clear the old gradient
"""

import torch


x = torch.tensor([1.,2.,3.,4.])
y_true = torch.tensor([2.,4.,6.,8.])

w = torch.tensor(0.0, requires_grad = True)

lr = 0.1
epochs = 10

for epochs in range(epochs):    #epoch is a label that'll help us to figure out on which step we are
    # step 1 = prediction
    y_pred = w * x

    # step 2 = loss

    loss = ((y_pred - y_true)**2).mean()

    #backward

    loss.backward()

    #gradient

    with torch.no_grad():      #with is used to temporarily change behaviour then return back
        w -= lr * w.grad

    print(f"Epoch : {epochs + 1},w:{w.item():.4f},loss:{loss.item():.4f}")