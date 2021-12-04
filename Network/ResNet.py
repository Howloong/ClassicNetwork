import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from utils.run_gpu04 import run_gpu04

run_gpu04()

import torch.nn as nn
import torch.optim
from torch.hub import load_state_dict_from_url
from utils.train_model import load_dataset, train_model

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# use multiple gpus
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def conv3x3(in_planes, out_planes=1, stride=1, ) -> nn.Conv2d:
    '''3x3的卷积层'''
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=(3, 3), stride=stride, padding=1,
                     bias=False, )


def conv1x1(inplanes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        inplanes,
        out_planes, kernel_size=1,
        stride=stride,
        bias=False
    )


class BasicBlock(nn.Module):
    # 在每个小block中，channel数不变
    def __init__(self, inplanes, planes, stride=1, downsample=None, ):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes=inplanes, out_planes=planes, stride=stride)
        # stride=2,且padding默认为1时，维度缩小为原来的1/2
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # stride=1，padding默认为1，维度不变。
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    expansion = 1

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        '''由于x的维度可能与out不同，无法相加，所以先将x进行下采样'''
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # 在每个小block中，channel数最后会变为原来的4倍

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)


class Resnet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        # 经过第一次conv后的输入通道数
        super(Resnet, self).__init__()
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, planes=64, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(block, planes=128, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes=256, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes=512, blocks=layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        # 最后一层的全连接，输入通道数取决于最后一个block的输出通道数。
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    '''Args:
        block:进行构造网络的基准模块，对于res18,res34来说是basicBlock，对于res50,res101,res152来说，
        是BottleneckBlock。
        planes:每个块的起始channel数
        blocks:每个块需要构造几个
    '''

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        '''如果stride为1，则维度不会变化。不为1说明维度发生了变化，
             为了能与x正常相加，需要将x进行下采样；
             对于第一个块是不需要下采样的(观察论文中给出的结构图，在第一个块中是没有虚线的。因为
             第一个块的输入与输出channel数相同，不需要进行下采样。
        '''
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet34(pretrained=False, **kwargs):
    model = Resnet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet34-b627a593.pth", )
    return model


net = Resnet(BasicBlock, [3, 4, 6, 3])
optimizer = torch.optim.Adam(net.parameters())
criterion = nn.CrossEntropyLoss()
batch_size = 500
train_loader, test_loader = load_dataset(batch_size)
train_model(model=net, epoch=10, batch_size=batch_size, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer)
