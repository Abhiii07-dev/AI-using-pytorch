import torch

#INput x

x  = torch.tensor([1.,2.,3.,4.])


# y

y = torch.tensor([2.,4.,6.,8.])

# Wrong guessing for w and b

w = torch.tensor(0.0)
b = torch.tensor(0.0)

y_pred = w * x + b

loss = ((y_pred - y) ** 2).mean()

# print(f"Actual Data:\n{y}")
# print(f"Predicted Data:\n{y_pred}")
# print(f"loss:\n{loss}")





