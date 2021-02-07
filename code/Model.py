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

#PNASNet
class SepConv(nn.Module):
    '''Separable Convolution.'''
    def __init__(self, in_planes, out_planes, kernel_size, stride):
        super(SepConv, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes,
                               kernel_size, stride,
                               padding=(kernel_size-1)//2,
                               bias=False, groups=in_planes)
        self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn1(self.conv1(x))


class CellA(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellA, self).__init__()
        self.stride = stride
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        global embeddings
        y1 = self.sep_conv1(x)
        y2 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y2 = self.bn1(self.conv1(y2))
        y = F.relu(y1+y2)
        embeddings.append(y)
        return y

class CellB(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(CellB, self).__init__()
        self.stride = stride
        # Left branch
        self.sep_conv1 = SepConv(in_planes, out_planes, kernel_size=7, stride=stride)
        self.sep_conv2 = SepConv(in_planes, out_planes, kernel_size=3, stride=stride)
        # Right branch
        self.sep_conv3 = SepConv(in_planes, out_planes, kernel_size=5, stride=stride)
        if stride==2:
            self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(out_planes)
        # Reduce channels
        self.conv2 = nn.Conv2d(2*out_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        global embeddings
        # Left branch
        y1 = self.sep_conv1(x)
        y2 = self.sep_conv2(x)
        # Right branch
        y3 = F.max_pool2d(x, kernel_size=3, stride=self.stride, padding=1)
        if self.stride==2:
            y3 = self.bn1(self.conv1(y3))
        y4 = self.sep_conv3(x)
        # Concat & reduce channels
        b1 = F.relu(y1+y2)
        b2 = F.relu(y3+y4)
        y = torch.cat([b1,b2], 1)
        y = F.relu(self.bn2(self.conv2(y)))
        embeddings.append(y)
        return y

class PNASNet(nn.Module):
    def __init__(self, cell_type, num_cells, num_planes):
        super(PNASNet, self).__init__()
        self.in_planes = num_planes
        self.cell_type = cell_type

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_planes)

        self.layer1 = self._make_layer(num_planes, num_cells=6)
        self.layer2 = self._downsample(num_planes*2)
        self.layer3 = self._make_layer(num_planes*2, num_cells=6)
        self.layer4 = self._downsample(num_planes*4)
        self.layer5 = self._make_layer(num_planes*4, num_cells=6)

        self.linear = nn.Linear(num_planes*4, 10)

    def _make_layer(self, planes, num_cells):
        layers = []
        for _ in range(num_cells):
            layers.append(self.cell_type(self.in_planes, planes, stride=1))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def _downsample(self, planes):
        layer = self.cell_type(self.in_planes, planes, stride=2)
        self.in_planes = planes
        return layer

    def forward(self, x):
        global embeddings
        embeddings = []
        out = F.relu(self.bn1(self.conv1(x)))
        embeddings.append(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.layer5(out)

        out = F.avg_pool2d(out, 8)
        embeddings.append(out)

        out = self.linear(out.view(out.size(0), -1))
        output = [embeddings]

        output.append(out)

        return output


def PNASNetA():
    return PNASNet(CellA, num_cells=6, num_planes=44)

def PNASNetB():
    return PNASNet(CellB, num_cells=6, num_planes=32)

#GoogLeNet
class Inception(nn.Module):
    def __init__(self, in_planes, n1x1, n3x3red, n3x3, n5x5red, n5x5, pool_planes):
        super(Inception, self).__init__()
        # 1x1 conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(in_planes, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(in_planes, n3x3red, kernel_size=1),
            nn.BatchNorm2d(n3x3red),
            nn.ReLU(True),
            nn.Conv2d(n3x3red, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv branch
        self.b3 = nn.Sequential(
            nn.Conv2d(in_planes, n5x5red, kernel_size=1),
            nn.BatchNorm2d(n5x5red),
            nn.ReLU(True),
            nn.Conv2d(n5x5red, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 1x1 conv branch
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_planes, kernel_size=1),
            nn.BatchNorm2d(pool_planes),
            nn.ReLU(True),
        )

    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3(x)
        y4 = self.b4(x)
        return torch.cat([y1,y2,y3,y4], 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )

        self.a3 = Inception(192,  64,  96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192,  96, 208, 16,  48,  64)
        self.b4 = Inception(512, 160, 112, 224, 24,  64,  64)
        self.c4 = Inception(512, 128, 128, 256, 24,  64,  64)
        self.d4 = Inception(512, 112, 144, 288, 32,  64,  64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        embedding = []

        out = self.pre_layers(x)
        embedding.append(out)

        out = self.a3(out)
        embedding.append(out)

        out = self.b3(out)
        embedding.append(out)

        out = self.maxpool(out)
        embedding.append(out)

        out = self.a4(out)
        embedding.append(out)

        out = self.b4(out)
        embedding.append(out)

        out = self.c4(out)
        embedding.append(out)

        out = self.d4(out)
        embedding.append(out)

        out = self.e4(out)
        embedding.append(out)

        out = self.maxpool(out)
        embedding.append(out)

        out = self.a5(out)
        embedding.append(out)

        out = self.b5(out)
        embedding.append(out)

        out = self.avgpool(out)
        embedding.append(out)

        output = [embedding]

        out = out.view(out.size(0), -1)
        out = self.linear(out)

        output.append(out)

        return output
