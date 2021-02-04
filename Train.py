#Importing modules

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from Model import *

#Preparing data
train_data = datasets.CIFAR10('data', train=True, download=True, transform = transforms.ToTensor())
train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train,batch_size=32)
val_loader =   DataLoader(val, batch_size=32)

#Loading Model
model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
# model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
# model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
# model = ResNet(Bottleneck, [3,4,23,3]) # Resnet101
# model = ResNet(Bottleneck, [3,4,36,3]) # Resnet152

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Defining optimiser
params = model.parameters()
optimiser = optim.SGD(params,lr=5e-2)
loss = nn.CrossEntropyLoss()

#Training
print("Training")

nb_epochs = 3
p = [0.0,0.0]

for epoch in range(nb_epochs):
    print(f'\nEpoch {epoch+1}', end = '\n')
    #track loss and accuracy
    losses = list()
    accuracies = list()

    if epoch == 15:
      print("Changing Dropout")
      p = [0.1,0.2]

    model.train() # because of dropout
    for batch in train_loader:
        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logits from all exit branches
        l = model(x,p)[-1][0]

        #2-objective function
        J = loss(l,y)

        #3-clean gradients
        model.zero_grad()

        #4-accumulate partial derivatives of J
        J.backward()

        #5-step in opposite direction of gradient
        optimiser.step()

        #6-record losses
        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
        break

    print('Training:')
    print(f'Loss: {torch.tensor(losses).mean():.2f}', end='\n')
    print(f'Accuracy: {torch.tensor(accuracies).mean():.2f}')

    #Reset losses and accuracies for validation
    losses = list()
    accuracies = list()

    model.eval()

    for batch in val_loader:

        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logits
        with torch.no_grad():
            l = model(x,p)[-1][0]

        #2-objective function
        J = loss(l,y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
        break

    print('Validation:')
    print(f'Loss: {torch.tensor(losses).mean():.2f}', end='\n')
    print(f'Accuracy: {torch.tensor(accuracies).mean():.2f}')

#Saving model
layer_wise_embeddings = []
n_epochs = 1

input,target = next(iter(train_loader))
input,target = input.to(device),target.to(device)
output = model(input)
output_shape = len(output[0])

n_layers = output_shape
n_targets = output[-1][0][0].shape[0] #Gets the output of the model, since it is returned as a list to account for branched networks


print("Number of layers: " + str(n_layers))
print("Number of targets: " + str(n_targets))

directory = "saved-models/"
torch.save(model.state_dict(), directory + 'ResNet'+str(n_layers)+'-CIFAR-10.pth')
