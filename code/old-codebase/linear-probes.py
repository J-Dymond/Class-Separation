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

for run in range(5):

    #Loading Model
    model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
    # model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
    # model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
    # model = BranchedResNet(BasicBlock, [2,2,2,2]) #ResNet18
    # model = BranchedResNet(BasicBlock, [3,4,6,3]) #ResNet34
    # model = BranchedResNet(Bottleneck, [3,4,6,3]) #ResNet50

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    #loading weights
    directory = "saved-models/"
    model_name = 'ResNet18-CIFAR-10'
    # model_name = 'ResNet34-CIFAR-10'
    # model_name = 'ResNet50-CIFAR-10'
    # model_name = 'BranchedResNet18-CIFAR-10'
    # model_name = 'BranchedResNet34-CIFAR-10'
    # model_name = 'BranchedResNet50-CIFAR-10'

    model_name = (model_name + '-' + str(run))

    model.load_state_dict(torch.load(directory + model_name + '.pth'))


    #Defining linear probe class
    class LinearClassifier(torch.nn.Module):
      def __init__(self, input_dimension):
        super().__init__()
        self.linear = torch.nn.Linear(input_dimension, 10)
      def forward(self,x):
        return F.dropout(self.linear(x),p=0.30)

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

    n_epochs = 10
    loss = nn.CrossEntropyLoss()

    for layer in range(n_layers):

      print('\nLayer: '+str(layer+1))

      classifier = classifiers[layer]
      params = classifier.parameters()
      classifier.to(device)
      optimiser = optim.SGD(params,lr=5e-2,momentum=0.9)

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

    save_directory = "linear-probe-values/accuracy/"
    save_string = save_directory+model_name+'.npy'
    np.save(save_string, linear_probes)

    save_directory = "linear-probe-values/loss/"
    save_string = save_directory+model_name+'.npy'
    np.save(save_string, linear_probes_loss)
