import torch
from torch import nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        global embedding

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        embedding.append(out)

        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        embedding.append(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):

        global embedding

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        embedding.append(out)

        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        embedding.append(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += self.shortcut(x)
        out = F.relu(out)
        embedding.append(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10, drop_prob=0.2, block_size=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # MNIST
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.dropout_conv = nn.Dropout2d(p=0.1)
        self.dropout = nn.Dropout2d(p=0.3)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, p=[0.0,0.0]):
        global embedding

        embedding = []
        branch_outputs = []

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        embedding.append(out)

        out = self.layer1(out)

        out = self.layer2(out)
        out = F.dropout(out, p=p[0], training=self.training)

        out = self.layer3(out)

        out = self.layer4(out)
        out = F.dropout(out, p=p[0], training=self.training)

        out = F.avg_pool2d(out, 4)
        embedding.append(out)

        out = out.view(out.size(0), -1)
        out = F.dropout(self.linear(out), p=p[1], training=self.training)

        output = [embedding]
        branch_outputs.append(out)

        output.append(branch_outputs)

        return output


class BranchedResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2,2,2,2], num_classes=10):
        super(BranchedResNet, self).__init__()
        self.in_planes = 64

        # self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False) # MNIST
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # CIFAR
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.branch_layer1 = nn.Linear(64*64*block.expansion, num_classes)
        self.branch_layer2 = nn.Linear(128*16*block.expansion, num_classes)
        self.branch_layer3 = nn.Linear(256*4*block.expansion, num_classes)

        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        global embedding
        embedding = []
        branch_outputs = []

        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        embedding.append(out)

        out = self.layer1(out)
        flat_out = F.avg_pool2d(out, 4)
        flat_out = flat_out.view(flat_out.size(0), -1)
        branch_outputs.append(self.branch_layer1(flat_out))

        out = self.layer2(out)
        # Can have branch after block 2
        # flat_out = F.avg_pool2d(out, 4)
        # flat_out = flat_out.view(flat_out.size(0), -1)
        # branch_outputs.append(self.branch_layer2(flat_out))

        out = self.layer3(out)
        flat_out = F.avg_pool2d(out, 4)
        flat_out = flat_out.view(flat_out.size(0), -1)
        branch_outputs.append(self.branch_layer3(flat_out))

        out = self.layer4(out)

        out = F.avg_pool2d(out, 4)
        embedding.append(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        output = [embedding]
        branch_outputs.append(out)

        output.append(branch_outputs)

        return output
