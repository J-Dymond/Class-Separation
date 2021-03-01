#Importing modules
import os
import torch
from torch import nn
from torch import optim
from torchvision import datasets,transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import numpy as np
import argparse
from Model import *

#arguments for running

parser = argparse.ArgumentParser()
parser.add_argument("target_directory",help="Name of directory to save data", type=str)
parser.add_argument("-m","--model", help="Architecture to be used",type=str,default='ResNet18')
parser.add_argument("-r","--runs", help="Number of runs of the experiment to do",type=int,default=5)
parser.add_argument("-e","--epochs", help="Number of epochs to run experiment for",type=int,default=200)
parser.add_argument("-b","--batch_size", help="Batch size for training",type=int,default=128)
parser.add_argument("-lr","--learning_rate", help="Learning rate for training",type=int,default=1e-2)
args = parser.parse_args()

print('arguments passed:')
print("target_directory: " + args.target_directory)
print("Architecture: " + args.model)
print("Epochs: " + str(args.epochs))
print("Runs: " + str(args.runs))
print("Batch size: " + str(args.batch_size))
print("Learning Rate: "+str(args.learning_rate))

try:
    # Create target Directory
    os.mkdir("trained-models/"+args.model)
    print("Directory: " , "trained-models/"+args.model ,  " Created ")
except FileExistsError:
    print("Directory: " , "trained-models/"+args.model ,  " already exists")

try:
    save_directory = "trained-models/"+args.model+"/"+args.target_directory+"/"
    os.mkdir(save_directory)
    print("Saving data to: " , save_directory)

except FileExistsError:
    print(save_directory, "Already exists...")
    for run in range(100):
        try:
            save_directory = "trained-models/"+args.model+"/run"+str(run)+"/"
            os.mkdir(save_directory)
            print("Instead saving data to: " , save_directory)
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
train_loader = DataLoader(train_data,batch_size=args.batch_size)
val_loader =   DataLoader(test_data, batch_size=args.batch_size)

train_losses = np.zeros((args.runs,args.epochs))
train_accs = np.zeros((args.runs,args.epochs))

val_losses = np.zeros((args.runs,args.epochs))
val_accs = np.zeros((args.runs,args.epochs))

checkpoint_metrics = np.zeros((args.runs,3))

for run in range(args.runs):

    print("Run:" + str(run+1))

    if args.model == 'ResNet18':
        model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18
    elif args.model == 'ResNet34':
        model = ResNet(BasicBlock, [3,4,6,3]) # Resnet34
    elif args.model == 'ResNet50':
        model = ResNet(Bottleneck, [3,4,6,3]) # Resnet50
    else:
        model = ResNet(BasicBlock, [2,2,2,2]) # Resnet18

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
    optimiser = optim.SGD(params, lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=200)
    loss = nn.CrossEntropyLoss()

    #Training
    print("Training:")
    best_accuracy = -1
    nb_epochs = args.epochs
    epoch_gradients_L1 = list()
    epoch_gradients_L2 = list()

    train_accuracy =  np.zeros(args.epochs)
    val_accuracy = np.zeros(args.epochs)
    train_loss= np.zeros(args.epochs)
    val_loss = np.zeros(args.epochs)

    for epoch in range(nb_epochs):
        #track loss and accuracy
        losses = list()
        accuracies = list()
        model.train() # because of dropout
        batch_gradients_L1 = torch.zeros(n_conv_layers)
        batch_gradients_L2 = torch.zeros(n_conv_layers)
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
                flattened_gradients = torch.flatten(param.grad)
                batch_gradients_L1[layer] = batch_gradients_L1[layer] + torch.linalg.norm(flattened_gradients,ord=1)/flattened_gradients.shape[0]
                batch_gradients_L2[layer] = batch_gradients_L1[layer] + torch.linalg.norm(flattened_gradients,ord=2)/flattened_gradients.shape[0]
                layer = layer + 1


            #5-step in opposite direction of gradient
            optimiser.step()

            losses.append(J.item())
            accuracies.append(y.eq(l.detach().argmax(dim=1)).float().mean())
            break

        train_accuracy[epoch] = torch.tensor(accuracies).mean()
        train_loss[epoch] = torch.tensor(losses).mean()

        epoch_gradient_L1 = batch_gradients_L1/len(train_loader)
        epoch_gradients_L1.append(np.array(epoch_gradient_L1.detach().cpu().numpy()))

        epoch_gradient_L2 = batch_gradients_L2/len(train_loader)
        epoch_gradients_L2.append(np.array(epoch_gradient_L2.detach().cpu().numpy()))


        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Training Loss: {train_loss[epoch]:.2f}', end=', ')
        print(f'Training Accuracy: {train_accuracy[epoch]:.2f}')

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
            break

        val_loss[epoch] = torch.tensor(losses).mean()
        val_accuracy[epoch] = torch.tensor(accuracies).mean()

        print(f'Epoch {epoch+1}', end = ', ')
        print(f'Validation Loss: {torch.tensor(losses).mean():.2f}', end=', ')
        print(f'Validation Accuracy: {torch.tensor(accuracies).mean():.2f}')

        if val_accuracy[epoch] > best_accuracy:
            torch.save(model.state_dict(), model_directory + 'best-ResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')
            best_accuracy = val_accuracy[epoch]

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
    save_string = gradient_directory+'ResNet'+str(n_layers)+'-'+str(run)+'-L1.npy'
    np.save(save_string, numpy_epoch_gradients_L1)

    numpy_epoch_gradients_L2 = np.array(epoch_gradients_L2)
    save_string = gradient_directory+'ResNet'+str(n_layers)+'-'+str(run)+'-L2.npy'
    np.save(save_string, numpy_epoch_gradients_L2)

    #Saving model
    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))


    torch.save(model.state_dict(), model_directory + 'epoch'+str(epoch)+'-ResNet'+str(n_layers)+'-CIFAR-10-'+str(run)+'.pth')

    val_accs[run,:] = val_accuracy
    val_losses[run,:] = val_loss
    train_accs[run,:] = train_accuracy
    train_losses[run,:] = train_loss
    checkpoint_metrics[run,:] = np.array([epoch,best_accuracy,optimiser.param_groups[0]['lr']])

save_string = metric_directory+'val-accuracy-ResNet'+str(n_layers)+'.npy'
np.save(save_string, val_accs)

save_string = metric_directory+'train-accuracy-ResNet'+str(n_layers)+'.npy'
np.save(save_string, train_accs)

save_string = metric_directory+'val-losses-ResNet'+str(n_layers)+'.npy'
np.save(save_string, val_losses)

save_string = metric_directory+'train-losses-ResNet'+str(n_layers)+'.npy'
np.save(save_string, train_losses)

save_string = metric_directory+'checkpoint-metrics-ResNet'+str(n_layers)+'.npy'
np.save(save_string, checkpoint_metrics)
