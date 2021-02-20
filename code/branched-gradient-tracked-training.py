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
parser.add_argument("-m","--model", help="Backbone architecture to be used",type=str,default='ResNet18')
parser.add_argument("-w","--branch_weightings", nargs="+",
    help="How to weight the branch losses. Format: a b c d -> a+b+c+d should equal 1.0. Default:[0.2,0.0,0.3,0.5]. Use 0.0 to deactivate branch.",
    type=float,default=[0.2,0.0,0.3,0.5])
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do",type=int,default=5)
parser.add_argument("-e","--epochs", help="Number of epochs to run experiment for",type=int,default=200)
parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=int,default=1e-2)
args = parser.parse_args()

print('arguments passed:')
print("Architecture: " + args.model)
print("Branch Weightings: " + str(args.branch_weightings))
print("Epochs: " + str(args.epochs))
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))

#prepare directories for saving data
try:
    # Create target Directory
    os.mkdir("trained-models/Branched"+args.model)
    print("Directory " , "trained-models/Branched"+args.model ,  " Created ")
except FileExistsError:
    print("Directory " , "trained-models/Branched"+args.model ,  " already exists")

for run in range(100):
    try:
        save_directory = "trained-models/Branched"+args.model+"/run"+str(run)+"/"
        os.mkdir(save_directory)
        print("Saving data to: " , save_directory)
        break

    except FileExistsError:
        continue

metric_directory = save_directory+"/metrics/"
os.mkdir(metric_directory)

model_directory = save_directory+"/saved-models/"
os.mkdir(model_directory)

gradient_directory = save_directory+"/gradient-values/"
os.mkdir(gradient_directory)


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

branch_weights = args.branch_weightings

train_losses = np.zeros((args.runs,args.epochs))
val_losses = np.zeros((args.runs,args.epochs))

branch_train_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_accs = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_train_losses = np.zeros((args.runs,len(branch_weights),args.epochs))
branch_val_losses = np.zeros((args.runs,len(branch_weights),args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):

    print("\nRun:" + str(run+1))

    if args.model == 'ResNet18':
        model = BranchedResNet(BasicBlock, [2,2,2,2]) # Resnet18
    elif args.model == 'ResNet34':
        model = BranchedResNet(BasicBlock, [3,4,6,3]) # Resnet34
    elif args.model == 'ResNet50':
        model = BranchedResNet(Bottleneck, [3,4,6,3]) # Resnet50
    else:
        model = BranchedResNet(BasicBlock, [2,2,2,2]) # Resnet18

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
    optimiser = optim.SGD(params, lr=args.learning_rate,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    loss = nn.CrossEntropyLoss()

    #Training

    best_accuracy = -1.0
    nb_epochs = args.epochs
    epoch_gradients_L1 = list()
    epoch_gradients_L2 = list()

    for epoch in range(nb_epochs):
        #track loss and accuracy
        losses = list()
        accuracies = list()

        #and for the branches
        branch_losses = list()
        for i in range(len(branch_weights)):
          branch_losses.append(list())
        branch_accuracies = list()
        for i in range(len(branch_weights)):
          branch_accuracies.append(list())

        model.train() # because of dropout
        batch_gradients_L1 = torch.zeros(n_conv_layers)
        batch_gradients_L2 = torch.zeros(n_conv_layers)

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
            E4 = l[3]

            #losses, all exits have same target
            L1 = loss(E1,y)
            L2 = loss(E2,y)
            L3 = loss(E3,y)
            L4 = loss(E4,y)

            #2-objective function
            J = branch_weights[0]*L1 + branch_weights[1]*L2 + branch_weights[2]*L3 + branch_weights[3]*L4

            #3-clean gradients
            model.zero_grad()

            #4-accumulate partial derivatives of J
            J.backward()

            layer = 0
            for name, param in model.named_parameters():
              if "conv" in name:
                batch_gradients_L1[layer] = batch_gradients_L1[layer] + torch.sum(torch.abs(param.grad))/torch.flatten(param).shape[0]
                batch_gradients_L2[layer] = batch_gradients_L2[layer] + torch.sum(torch.abs(param.grad))/torch.flatten(param).shape[0]
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
            branch_losses[3].append(L4.item())

            branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
            branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
            branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())
            branch_accuracies[3].append(y.eq(E4.detach().argmax(dim=1)).float().mean())
            break

        epoch_gradient_L1 = batch_gradients_L1/len(train_loader)
        epoch_gradients_L1.append(np.array(epoch_gradient_L1.detach().cpu().numpy()))

        epoch_gradient_L2 = batch_gradients_L2/len(train_loader)
        epoch_gradients_L2.append(np.array(epoch_gradient_L2.detach().cpu().numpy()))

        train_losses[run,epoch] = torch.tensor(losses).mean()

        print(f'\n\nEpoch {epoch+1}', end = '\n')
        print('Training:')
        print(f'Total Loss: {train_losses[run,epoch]:.2f}', end='\n')
        print('Exit Losses: ')
        for i in range(len(branch_weights)):
            branch_train_losses[run,i,epoch] = torch.tensor(branch_losses[i]).mean()
            print(f'Exit {(i+1)}: {branch_train_losses[run,i,epoch]:.2f}', end=', ')
        print('\nExit accuracies: ')
        for i in range(len(branch_weights)):
            branch_train_accs[run,i,epoch] = torch.tensor(branch_accuracies[i]).mean()
            print(f'Exit {(i+1)}: {branch_train_accs[run,i,epoch]:.2f}', end=', ')

        #Reset losses
        losses = list()
        accuracies = list()
        #and for the branches
        branch_losses = list()
        for i in range(len(branch_weights)):
          branch_losses.append(list())
        branch_accuracies = list()
        for i in range(len(branch_weights)):
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
                E4 = l[3]

            #losses, all exits have same target
            L1 = loss(E1,y)
            L2 = loss(E2,y)
            L3 = loss(E3,y)
            L4 = loss(E4,y)

            #2-objective function
            J = branch_weights[0]*L1 + branch_weights[1]*L2 + branch_weights[2]*L3 + branch_weights[3]*L4

            losses.append(J.item())

            #also for branches
            branch_losses[0].append(L1.item())
            branch_losses[1].append(L2.item())
            branch_losses[2].append(L3.item())
            branch_losses[3].append(L4.item())

            branch_accuracies[0].append(y.eq(E1.detach().argmax(dim=1)).float().mean())
            branch_accuracies[1].append(y.eq(E2.detach().argmax(dim=1)).float().mean())
            branch_accuracies[2].append(y.eq(E3.detach().argmax(dim=1)).float().mean())
            branch_accuracies[3].append(y.eq(E4.detach().argmax(dim=1)).float().mean())
            break

        val_losses[run,epoch] = torch.tensor(losses).mean()

        print('\nValidation:')
        print(f'Total Loss: {val_losses[run,epoch]:.2f}', end='\n')
        print('Exit Losses: ')
        for i in range(len(branch_weights)):
            branch_val_losses[run,i,epoch] = torch.tensor(branch_losses[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_losses[run,i,epoch]:.2f}', end=', ')
        print('\nExit accuracies: ')
        for i in range(len(branch_weights)):
            branch_val_accs[run,i,epoch] = torch.tensor(branch_accuracies[i]).mean()
            print(f'Exit {(i+1)}: {branch_val_accs[run,i,epoch]:.2f}', end=', ')

        val_acc = max(branch_val_accs[run,:,epoch])

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-BranchedResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_acc

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
    numpy_epoch_gradients_L1 = np.array(epoch_gradients_L1)
    save_string = gradient_directory+'BranchedResNet'+str(n_layers)+'-'+str(run)+'-L1.npy'
    np.save(save_string, numpy_epoch_gradients_L1)

    numpy_epoch_gradients_L2 = np.array(epoch_gradients_L2)
    save_string = gradient_directory+'BranchedResNet'+str(n_layers)+'-'+str(run)+'-L2.npy'
    np.save(save_string, numpy_epoch_gradients_L2)

    #Saving model
    print("\n\nSaving model..")
    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))

    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-BranchedResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')

    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

#Saving all tracked metrics for analysis

save_string = metric_directory+'val-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'branch-val-accuracies-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_val_accs)

save_string = metric_directory+'branch-train-accuracies-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_train_accs)

save_string = metric_directory+'branch-train-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_train_losses)

save_string = metric_directory+'branch-val-losses-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_val_losses)

save_string = metric_directory+'checkpoint-metrics-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, checkpoint_metrics)

save_string = metric_directory+'branch-weights-BranchedResNet'+str(n_layers)+'.npy'
np.save(save_string, branch_weights)
