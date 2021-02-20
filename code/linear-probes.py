#Importing modules
import os
import argparse

import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
from Model import *

#Arguments for running
parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Directory to run class separation code in. Format: '../trained-models/*model*/Runx/'", type=str)
parser.add_argument("-m","--model", help="Backbone architecture to be used",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do. Must be less than or equal to number of trained models.",type=int,default=5)
parser.add_argument("-b","--batch_size", help="Batch size for training.",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for probe training",type=int,default=1e-2)
parser.add_argument("-e","--epochs",help="Number of epochs for linear probe training",type=int,default=10)
args = parser.parse_args()

print('arguments passed:'+args.target_directory)
print("Target")
print("Architecture: " + args.model)
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))
print("Learning Rate: "+str(args.learning_rate))

directory = args.target_directory

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir(directory+"linear-probe-values/")
    print("Directory: " , directory+"linear-probe-values/" ,  " Created ")
except FileExistsError:
    print("Directory: " ,directory+"linear-probe-values/" ,  " already exists")

try:
    # Create target Directory
    os.mkdir("trained-models/"+args.model)
    print("Directory: " , directory+"linear-probe-values/accuracy" ,  " Created ")
except FileExistsError:
    print("Directory: " , directory+"linear-probe-values/accuracy" ,  " already exists")

try:
    # Create target Directory
    os.mkdir("trained-models/"+args.model)
    print("Directory: " , directory+"linear-probe-values/loss" ,  " Created ")
except FileExistsError:
    print("Directory: " , directory+"linear-probe-values/loss" ,  " already exists")



save_directory = directory+"linear-probe-values/"
model_directory = directory+"saved-models/"

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

for run in range(args.runs):
    print("Run: " + str(run))

    #defining model

    if args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
    elif args.model == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
    elif args.model == 'BranchedResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2]) # Resnet18
    elif args.model == 'BranchedResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3]) # Resnet34
    elif args.model == 'BranchedResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3]) # Resnet50
    else:
        model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18

    #putting to GPU and loading weights
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model_name = "best-"+args.model+'-CIFAR-10-'+str(run)+'.pth'
    model.load_state_dict(torch.load(model_directory + model_name))


    #Defining linear probe class
    class LinearClassifier(torch.nn.Module):
      def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, 10)
      def forward(self,x):
        return F.dropout(self.linear(x),p=0.15)

    #Get model size
    n_epochs = 1

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    embeddings = model(input)[0][:]
    output_shape = len(embeddings)

    n_layers = output_shape
    n_targets = model(input)[-1][-1][0].shape[0] #Gets the output of the model, since it is returned as a list to account for branched networks

    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))

    #Generate classifiers
    classifiers = []

    for i in range (0, n_layers):
        classifier_dimension = torch.flatten(embeddings[i][0]).shape[0]
        classifiers.append(LinearClassifier(classifier_dimension))


    #Train classifiers

    linear_probes = []
    linear_probes_loss = []

    n_epochs = args.epochs
    loss = nn.CrossEntropyLoss()

    for layer in range(n_layers):

      print('\nLayer: '+str(layer+1))

      classifier = classifiers[layer]
      params = classifier.parameters()
      classifier.to(device)
      optimiser = optim.SGD(params,lr=args.learning_rate,momentum=0.9)

      for epoch in range(n_epochs):
        print('\nEpoch: '+str(epoch+1))

        losses = list()
        accuracies = list()

        classifier.train()
        for batch in train_loader:

            x,y = batch
            x,y = x.to(device),y.to(device)

            with torch.no_grad():
              batch_embeddings = model(x)[0]

            layer_input = torch.flatten(batch_embeddings[layer],start_dim=1)
            l = classifier(layer_input)

            #2-objective function
            J = loss(l,y)

            #3-clean gradients
            classifier.zero_grad()

            #4-accumulate partial derivatives of J
            J.backward()

            #5-step in opposite direction of gradient
            optimiser.step()

        losses.append(J.item())
        accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
        print('Training:')
        print(f'Loss: {torch.tensor(losses).mean():.2f}', end='\n')
        print(f'Accuracy: {torch.tensor(accuracies).mean():.2f}')

        #Reset losses and accuracies for validation
        val_losses = list()
        val_accuracies = list()

        classifier.eval()

        for batch in val_loader:

            x,y = batch
            x,y = x.to(device),y.to(device)

            with torch.no_grad():
              batch_embeddings = model(x)[0]

            layer_input = torch.flatten(batch_embeddings[layer],start_dim=1)
            with torch.no_grad():
              l = classifier(layer_input)

            #2-objective function
            J = loss(l,y)

            val_losses.append(J.item())
            val_accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())

        val_loss = torch.tensor(val_losses).mean()
        val_acc = torch.tensor(val_accuracies).mean()

        print('Validation:')
        print(f'Loss: {val_loss:.2f}', end='\n')
        print(f'Accuracy: {val_acc:.2f}')

      print()

      linear_probes.append(val_acc.detach().cpu().numpy())
      linear_probes_loss.append(val_loss.detach().cpu().numpy())

    linear_probes = np.array(linear_probes)
    linear_probes_loss = np.array(linear_probes_loss)

    save_string = save_directory+model_name+'.npy'
    np.save(save_string, linear_probes)

    save_string = save_directory+model_name+'.npy'
    np.save(save_string, linear_probes_loss)
