"""
#DATALOADERSSSSSS........


dataset -> 1 sample
data loader  0> group into smol batches
model -> learn batch    
"""


"""
Overview of the Project

it is a dataset which contains black and white images and each image consists of 1 handritten number(0-9)
input - image
output - handwritten number

THIS IS CALLED MULTICLASS PROBLEM

#flow chart

input image -> convert image to number -> give it to the model -> suppose it'll give 10 output(0 - 9) -> have to choose the highest possible number


"""


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device",device)

transform = transforms.ToTensor()

train_data = MNIST(
    root = "data",
    train = True,
    download = True,
    transform = transform
       
)

test_data = MNIST(
    root = "data",
    train = False,
    download = True,
    transform = transform
       
)

train_loader = DataLoader(
    train_data,
    batch_size = 64,
    shuffle = True,
    drop_last = True
)

test_loader = DataLoader(
    test_data,
    batch_size = 64,
    shuffle = False,
    drop_last = True
)

class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(),

            nn.Linear(128,64),
            nn.ReLU(),

            nn.Linear(64,10)
        )

    def forward(self,x):
        return self.net(x)


model = DigitModel().to(device)

loss_fn = nn.CrossEntropyLoss()
optimiser = optim.Adam(model.parameters() , lr = 0.001)

epochs  = 5 

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for image,labels in train_loader:
        images = image.view(image.size(0),-1).to(device)
        labels = labels.to(device)


        optimiser.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs,labels)
        loss.backward()
        optimiser.step()


        total_loss += loss.item()

    print(f"Epoch {epoch +1}/{epochs}| Loss : {total_loss:.4f}")


model.eval()
correct = 0
total = 0


with torch.no_grad():
    for images,labels in test_loader:
        images = image.view(image.size(0),-1).to(device)
        labels = labels.to(device)

        outputs = model(images)
        predictions = outputs.argmax(dim = 1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = 100 * correct/total
print(f"Total accuracy : {accuracy:.2f}%")

torch.save(model.state_dict(),"mnist_model.pth")
print("Model saved as mnist_model.pth")

index =  80
image,true_label = test_data[index]

plt.imshow(image.squeeze(),cmap = "grey")
plt.title(f"Actual label : {true_label}")
plt.axis("off")
plt.show()


image_flat = image.view(1,-1).to(device)

with torch.no_grad():
    output = model(image_flat)
    predicted_label = output.argmax(dim=1).item()


print("User picked image index",index)
print("Actual label",true_label)
print("Model prediction",predicted_label)