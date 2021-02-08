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
# model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
# model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
# model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
# model = BranchedResNet(BasicBlock, [2,2,2,2]) #ResNet18
# model = BranchedResNet(BasicBlock, [2,4,6,3]) #ResNet34
# model = BranchedResNet(Bottleneck, [3,4,6,3]) #ResNet50
model = GoogLeNet()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Number of convolutional layers to be tracked
layer = 0
previous_name = 'a3'
for name, param in model.named_parameters():
  if 'weight' in name:
    if 'linear' not in name and 'pre' not in name:
      if name[:2] == previous_name[:2]:
        previous_name = name
      else:
        layer = layer + 1
        n_conv_layers = layer
        previous_name = name

print("Number of convolutional layers: " + str(n_conv_layers))

#Defining optimiser
params = model.parameters()
optimiser = optim.SGD(params,lr=5e-2)
loss = nn.CrossEntropyLoss()

#Training
print("Training:")

nb_epochs = 50
epoch_gradients = list()
for epoch in range(nb_epochs):
    #track loss and accuracy
    losses = list()
    accuracies = list()
    model.train() # because of dropout
    batch_gradients = torch.zeros(n_conv_layers)
    for batch in train_loader:
        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logits
        l = model(x)[-1][0]
        #2-objective function
        J = loss(l,y)
        #3-clean gradients
        model.zero_grad()
        #4-accumulate partial derivatives of J
        J.backward()

        layer = 0
        previous_name = 'a3'
        for name, param in model.named_parameters():
          if 'weight' in name:
            if 'linear' not in name and 'pre' not in name:
              if name[:2] == previous_name[:2]:
                batch_gradients[layer] = batch_gradients[layer] + torch.sum(torch.abs(param.grad))
                previous_name = name
              else:
                layer = layer + 1
                batch_gradients[layer] = batch_gradients[layer] + torch.sum(torch.abs(param.grad))
                n_conv_layers = layer
                previous_name = name


        #5-step in opposite direction of gradient
        optimiser.step()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    epoch_gradient = batch_gradients/len(train_loader)
    epoch_gradients.append(np.array(epoch_gradient.detach().cpu().numpy()))


    print(f'Epoch {epoch+1}', end = ', ')
    print(f'Training Loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'Training Accuracy: {torch.tensor(accuracies).mean():.2f}')

    losses = list()
    accuracies = list()
    model.eval()

    for batch in val_loader:

        x,y = batch
        x,y = x.to(device),y.to(device)
        #x: b x 1 x 28 x 28
        b = x.size(0)
        # x = x.view(b,-1)
        #1-forward pass - get logites
        with torch.no_grad():
            l = model(x)[-1][0]
        #2-objective function
        J = loss(l,y)

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

    print(f'Epoch {epoch+1}', end = ', ')
    print(f'Validation Loss: {torch.tensor(losses).mean():.2f}', end=', ')
    print(f'Validation Accuracy: {torch.tensor(accuracies).mean():.2f}')

#Saving model
layer_wise_embeddings = []
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

# save_string = numpy_directory+'ResNet'+str(n_layers)+'.npy'
save_string = numpy_directory+'GoogLeNet.npy'

np.save(save_string, numpy_epoch_gradients)

#Saving model
print("Number of layers: " + str(n_layers))
print("Number of targets: " + str(n_targets))

directory = "saved-models/"
# torch.save(model.state_dict(), directory + 'ResNet'+str(n_layers)+'-CIFAR-10.pth')
torch.save(model.state_dict(), directory+ 'GoogLeNet-CIFAR-10.pth')
