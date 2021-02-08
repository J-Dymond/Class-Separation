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
model = BranchedResNet(BasicBlock, [2,2,2,2]) #ResNet18
# model = BranchedResNet(BasicBlock, [2,4,6,3]) #ResNet34
# model = BranchedResNet(Bottleneck, [3,4,6,3]) #ResNet50

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Number of convolutional layers to be tracked
n_conv_layers = 0
for name, param in model.named_parameters():
  if 'conv' in name:
    n_conv_layers = n_conv_layers + 1

print("Number of convolutional layers: " + str(n_conv_layers))

#Defining optimiser
params = model.parameters()
optimiser = optim.SGD(params,lr=5e-2)
loss = nn.CrossEntropyLoss()

#Training

nb_epochs = 50
epoch_gradients = list()
for epoch in range(nb_epochs):
    #track loss and accuracy
    losses = list()
    accuracies = list()

    #and for the branches
    branch_losses = list()
    for i in range(3):
      branch_losses.append(list())
    branch_accuracies = list()
    for i in range(3):
      branch_accuracies.append(list())

    model.train() # because of dropout
    batch_gradients = torch.zeros(n_conv_layers)

    for batch in train_loader:
        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logits from all exit branches
        l = model(x)[-1]

        E1 = l[0]
        E2 = l[1]
        E3 = l[2]

        #losses, all exits have same target
        L1 = loss(E1,y)
        L2 = loss(E2,y)
        L3 = loss(E3,y)

        #2-objective function
        J = 0.3*L1 + 0.2*L2 + 0.5*L3

        #3-clean gradients
        model.zero_grad()

        #4-accumulate partial derivatives of J
        J.backward()
        layer = 0
        for name, param in model.named_parameters():
          if "conv" in name:
            batch_gradients[layer] = batch_gradients[layer] + torch.sum(torch.abs(param.grad))
            layer = layer + 1

        #5-step in opposite direction of gradient
        optimiser.step()

        #6-record losses
        losses.append(J.item())
        accuracies.append(y.eq(E3.detach().argmax(dim=1)).float().mean())

        #also for branches
        branch_losses[0].append(L1.item())
        branch_losses[1].append(L2.item())
        branch_losses[2].append(L3.item())

        branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
        branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
        branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())

    epoch_gradient = batch_gradients/len(train_loader)
    epoch_gradients.append(np.array(epoch_gradient.detach().cpu().numpy()))

    print(f'\nEpoch {epoch+1}', end = '\n')
    print('Training:')
    print(f'Total Loss: {torch.tensor(losses).mean():.2f}', end='\n')
    print(f'Final Exit Accuracy: {torch.tensor(accuracies).mean():.2f}')
    print('Exit Losses: ')
    print(f'Exit 1: {torch.tensor(branch_losses[0]).mean():.2f}', end=', ')
    print(f'Exit 2: {torch.tensor(branch_losses[1]).mean():.2f}', end=', ')
    print(f'Final Exit: {torch.tensor(branch_losses[2]).mean():.2f}', end='\n')
    print('Exit accuracies: ')
    print(f'Exit 1: {torch.tensor(branch_accuracies[0]).mean():.2f}', end=', ')
    print(f'Exit 2: {torch.tensor(branch_accuracies[1]).mean():.2f}', end=', ')
    print(f'Final Exit: {torch.tensor(branch_accuracies[2]).mean():.2f}', end='\n')

    #Reset losses
    losses = list()
    accuracies = list()
    #and for the branches
    branch_losses = list()
    for i in range(3):
      branch_losses.append(list())
    branch_accuracies = list()
    for i in range(3):
      branch_accuracies.append(list())
    model.eval()

    for batch in val_loader:

        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logits
        with torch.no_grad():
            l = model(x)[-1]


            E1 = l[0]
            E2 = l[1]
            E3 = l[2]

        L1 = loss(E1,y)
        L2 = loss(E2,y)
        L3 = loss(E3,y)

        #2-objective function
        J = 0.2*L1 + 0.5*L2 + 0.3*L3

        losses.append(J.item())
        accuracies.append(y.eq(E3.detach().argmax(dim=1)).float().mean())

        #also for branches
        branch_losses[0].append(L1.item())
        branch_losses[1].append(L2.item())
        branch_losses[2].append(L3.item())

        branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
        branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
        branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())

    print('Validation:')
    print(f'Total Loss: {torch.tensor(losses).mean():.2f}', end='\n')
    print(f'Final Exit Accuracy: {torch.tensor(accuracies).mean():.2f}')
    print('Exit Losses: ')
    print(f'Exit 1: {torch.tensor(branch_losses[0]).mean():.2f}', end=', ')
    print(f'Exit 2: {torch.tensor(branch_losses[1]).mean():.2f}', end=', ')
    print(f'Final Exit: {torch.tensor(branch_losses[2]).mean():.2f}', end='\n')
    print('Exit accuracies: ')
    print(f'Exit 1: {torch.tensor(branch_accuracies[0]).mean():.2f}', end=', ')
    print(f'Exit 2: {torch.tensor(branch_accuracies[1]).mean():.2f}', end=', ')
    print(f'Final Exit: {torch.tensor(branch_accuracies[2]).mean():.2f}', end='\n')

#Saving model
n_epochs = 1

input,target = next(iter(train_loader))
input,target = input.to(device),target.to(device)
output = model(input)
output_shape = len(output[0])
n_layers = output_shape
n_targets = output[-1][0][0].shape[0]

#Saving gradient values
numpy_epoch_gradients = np.array(epoch_gradients)
numpy_directory = "gradient-values/"
save_string = numpy_directory+'BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, numpy_epoch_gradients)

#Saving model
print("Number of layers: " + str(n_layers))
print("Number of targets: " + str(n_targets))

directory = "saved-models/"
torch.save(model.state_dict(), directory + 'BranchedResNet'+str(n_layers)+'-CIFAR-10.pth')
