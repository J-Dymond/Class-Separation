#importing modules
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

    #defining model
    model = ResNet(BasicBlock, [2,2,2,2]) #ResNet18
    # model = ResNet(BasicBlock, [3,4,6,3]) #ResNet34
    # model = ResNet(Bottleneck, [3,4,6,3]) #ResNet50
    # model = BranchedResNet(BasicBlock, [2,2,2,2]) #ResNet18
    # model = BranchedResNet(BasicBlock, [3,4,6,3]) #ResNet34
    # model = BranchedResNet(Bottleneck, [3,4,6,3]) #ResNet50
    # model = GoogLeNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    #loading weights
    directory = "saved-models/"
    model_name = 'ResNet18-CIFAR-10'
    # model_name = 'ResNet34-CIFAR-10'
    # model_name = 'ResNet50-CIFAR-10'
    # model_name = 'BranchedResNet18-CIFAR-10'
    # model_name = 'BranchedResNet34-CIFAR-10'
    # model_name = 'BranchedResNt50-CIFAR-10'
    # model_name = 'GoogLeNet-CIFAR-10'

    model_name = (model_name + '-' + str(run))

    model.load_state_dict(torch.load(directory + model_name + '.pth'))

    #defining embedding function
    def get_sim(a,b):
      dot = np.dot(a,b)
      abs_a = np.linalg.norm(a)
      abs_b = np.linalg.norm(b)

      return 1 - (dot/(abs_a*abs_b))


    def get_r(layer_embeddings, number_to_sample):

      sampled_embeddings = []
      i = 0
      for class_embedding in layer_embeddings:
        A = np.array(class_embedding)
        A = A[np.random.choice(A.shape[0], number_to_sample, replace=False)]
        sampled_embeddings.append(A)

      self_r = 0

      for x in range(10):
        r = 0
        for i in range(number_to_sample):
          for j in range(number_to_sample):
            r = get_sim(sampled_embeddings[x][i],sampled_embeddings[x][j])/(10*(number_to_sample)**2)
            self_r = self_r + r

      all_r = 0
      for x in range(10):
        for y in range(10):
          for i in range(number_to_sample):
            for j in range(number_to_sample):
              all_r = all_r + get_sim(sampled_embeddings[x][i],sampled_embeddings[y][j])/((10**2)*((number_to_sample)**2))


      R = 1 - self_r/all_r

      return R

    #getting embeddings
    layer_wise_embeddings = []
    n_epochs = 1

    input,target = next(iter(train_loader))
    input,target = input.to(device),target.to(device)
    output = model(input)
    output_shape = len(output[0])

    n_layers = output_shape

    n_targets = output[-1][-1][0].shape[0] #Gets the output of the model, since it is returned as a list to account for branched networks


    print("Number of layers: " + str(n_layers))
    print("Number of targets: " + str(n_targets))

    R_vals = []


    for epoch in range(n_epochs):

        for j in range(n_layers):
          layer_wise_embeddings.append(list())
          for i in range(n_targets):
            layer_wise_embeddings[-1].append([])

        # for larger networks
        batch_count = 0

        for batch in train_loader:
            x,y = batch
            x,y = x.to(device),y.to(device)

            batch_embeddings = model(x)[0][:]

            for i in range(len(y)):
              target = y[i].item()
              for layer in range(n_layers):
                embedding = torch.flatten(batch_embeddings[layer][i]).detach().cpu().numpy()
                layer_wise_embeddings[layer][target].append(embedding)

            # For larger networks -> RAM becomes an issue
            batch_count = batch_count + 1

            if (batch_count%10 == 0):
              batch_R_vals = []

              if (batch_count%100 == 0):
                print("Batch Number: " + str(batch_count))

              N=[]
              for i in range(10):
                N.append(len(layer_wise_embeddings[0][i]))
                sample_size = min(N)

              for i in range(n_layers):
                batch_R_vals.append(get_r(layer_wise_embeddings[i],sample_size))

              layer_wise_embeddings = []
              for j in range(n_layers):
                layer_wise_embeddings.append(list())
                for i in range(n_targets):
                  layer_wise_embeddings[-1].append([])

              R_vals.append(batch_R_vals)

    R_vals = np.array(R_vals)
    av_R_vals = np.average(R_vals,axis=0)
    print(av_R_vals)

    save_directory = "separation-values/"
    save_string = save_directory+model_name+'.npy'
    np.save(save_string, av_R_vals)
