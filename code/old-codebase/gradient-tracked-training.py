#Importing modules

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from Model import *

#Preparing data

#normalising data
transform_train = transforms.Compose([
transforms.RandomCrop(32, padding=4),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform = transform_train)
test_data = datasets.CIFAR10('data', train=False, download=True, transform = transform_test)

# train, val = random_split(train_data, [45000,5000])
train_loader = DataLoader(train_data,batch_size=128)
val_loader =   DataLoader(test_data, batch_size=128)

val_losses = []
val_accs = []

for run in range(5):

    print(run)

    #Loading Model
    model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
    # model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
    # model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
    # model = BranchedResNet(BasicBlock, [2,2,2,2]) #ResNet18
    # model = BranchedResNet(BasicBlock, [2,4,6,3]) #ResNet34
    # model = BranchedResNet(Bottleneck, [3,4,6,3]) #ResNet50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)
    output_shape = len(output[0])
    n_layers = output_shape
    n_targets = output[-1][0][0].shape[0]

    #Number of convolutional layers to be tracked
    n_conv_layers = 0
    for name, param in model.named_parameters():
      if 'conv' in name:
        n_conv_layers = n_conv_layers + 1

    print("Number of convolutional layers: " + str(n_conv_layers))

    #Defining optimiser and scheduler
    params = model.parameters()
    optimiser = optim.SGD(params, lr=1e-2,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    loss = nn.CrossEntropyLoss()

    #Training
    print("Training:")
    best_accuracy = 0.0
    nb_epochs = 200
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
            for name, param in model.named_parameters():
              if "conv" in name:
                batch_gradients[layer] = batch_gradients[layer] + torch.sum(torch.abs(param.grad))/torch.flatten(param).shape[0]
                layer = layer + 1


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

        val_loss = torch.tensor(losses).mean()
        val_acc = torch.tensor(accuracies).mean()

        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Validation Loss: {torch.tensor(losses).mean():.2f}', end=', ')
        print(f'Validation Accuracy: {torch.tensor(accuracies).mean():.2f}')

        if val_acc > best_accuracy:
            directory = "saved-models/"
            torch.save(model.state_dict(), directory + 'ResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_acc
            best_loss = val_loss

        scheduler.step()

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
    save_string = numpy_directory+'ResNet'+str(n_layers)+'-'+str(run)+'.npy'
    np.save(save_string, numpy_epoch_gradients)

    #Saving model
    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))

    # if val_acc > best_accuracy:
    #     directory = "saved-models/"
    #     torch.save(model.state_dict(), directory + 'ResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')

    val_accs.append(best_accuracy)
    val_losses.append(best_loss)

accuracy_directory = 'multi-run-metrics/accuracy-'
save_string = accuracy_directory+'ResNet'+str(n_layers)+'.npy'
np.save(save_string, val_accs)

loss_directory = 'multi-run-metrics/loss-'
save_string = loss_directory+'ResNet'+str(n_layers)+'.npy'
np.save(save_string, val_losses)
