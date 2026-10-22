"""
optimiser = see the gradient -> see ow much change is required -> updates the gradient -> removes the old gradients

# SGD: slowly will move to thr target ( best for basic)
torch.optim.SGD(model.paremeters() , lr = 0.01)

# Aadm : adapt the speed , best for fater learning , handles the  niosy data effectively
torch.optim.Adam(model.paremeters() , lr = 0.001) 



optimiser.zero_grad()
forward pass
loss calculation
loss.backward()
optimiser.step()


"""



import torch
import torch.nn as nn
import torch.optim as optim

#model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.Linear = nn.Linear(1,1)
    
    def forward(self,x):
        return self.Linear(x)

x = torch.tensor([[1.],[2.],[3.],[4.]])
y = torch.tensor([[2.],[4.],[6.],[8.]])

model = MyModel()

loss_fn = nn.MSELoss()

optimiser = optim.Adam(model.parameters() , lr = 0.1)

for epoch in range (10):
    optimiser.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred,y)
    loss.backward()
    optimiser.step()

    print(f"Epoch {epoch + 1} , loss : {loss.item():.4f}")


# Saving Model


torch.save(model.state_dict(),"linear_model.pth")


