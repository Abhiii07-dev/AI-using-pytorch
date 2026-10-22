import torch
import torch.nnas as nn




# nn.Linear(input_features,output_features)
# [height,weight,age]>risk score

# output = (input * weight) + bias




w = torch.tensor(0.0 , requires_grad=True)

class MyModel(nn.module):

    def __init__(self):
        super().__init__()         #Very important to inherit values


        

self.w = nn.Parameter(torch,tensor(0.0,))

# def forward(self,x)
# returnself.w*x
