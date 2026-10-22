"""
Backward Answers:
1. Who is responsble
2. how much

for this to activate:
    1. loss number(how wron we are)
    2. requires_grad
    3. computation Graph
"""
import torch





w = torch.tensor(2.0 , requires_grad= True)
x = torch.tensor(7.0)

y_pred = w*x
y_true = torch.tensor([10.0])

loss = ((y_pred - y_true) ** 2)
print(loss)
loss.backward()
print(w.grad)  #If w.grad > 0 = decrease w 
               # If w.grad < 0 = increase w

print("Prediction",y_pred.item())
print("loss",loss.item())
print("w.grad",w.grad.item())

#IF THE RWSULT IS NEGATIVE WE HAVE TO INCREASE THE VALUE OF W AND IF ITS POSITIVE THEN WE HAVE TO DECREASE IT