import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torchsummary as summary
import torchvision
from torchvision.transforms import transforms

from utils.train_model import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
'''定义参数'''
batch_size = 128
total_epoch = 10
lr = 0.001
class_num = 10

'''获取数据集'''
train_dataset = torchvision.datasets.CIFAR10(
    root='./data/', train=True, download=True,
    transform=transforms.ToTensor())

test_dataset = torchvision.datasets.CIFAR10(
    root='./data/', train=False,
    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        '''第一层卷积层，卷积核为3*3，通道数96，步距1，原始图像大小为32*32，有RGB三个通道'''
        '''feature map大小为 （32-3+2*0）/1+1=30，所以feature map大小为96*30*30'''
        self.conv1 = nn.Conv2d(3, 96, kernel_size=3, stride=1, padding=0)
        '''经过一次批归一化，将数据拉回到正态分布'''
        self.bn1 = nn.BatchNorm2d(96)
        '''第一层池化层，卷积层为3*3,步距为2，前一层的feature map大小为30*30，通道数为96'''
        '''经过第一层池化层后，得到的feature map大小为（30-
        3+2*0）/2+1=14,feature map的维度的个数为96*14*14'''
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        '''第二层卷积层，卷积核为3*3，步距为1，通道数为256，前一层的feature map大小为14*14，通道数96'''
        '''feature map 大小为 (14-3+2*0)/1+1=12'''
        self.conv2 = nn.Conv2d(96, 256, kernel_size=3, stride=1)
        '''标准化'''
        self.bn2 = nn.BatchNorm2d(256)
        '''第二层池化层，卷积核大小为3*3，步距为2，前一层的feature map大小为12*12，通道数256'''
        '''第二层池化层后，feature map大小为（12-3）/2+1=5'''
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        '''第三层池化层，卷积核大小为3*3，步距为1，padding=1，通道数384，前一层的feature map大小为5*5，通道数256'''
        '''第二层池化层后，feature map大小为（5-3+1*2）/1+1=5，feature map维度为384*5*5'''
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1, stride=1)
        '''第四层池化层，卷积核大小为3*3，步距为1，前一层的feature map大小为5*5，通道数384'''
        '''第二层池化层后，feature map大小为（5-3+1*2）/1+1=5，feature map维度为384*5*5'''
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1, stride=1)
        '''第五层卷积层，卷积核为3*3，通道数为384，步距为1，前一层的feature map的大小为5*5，通道数为384个'''
        '''这样经过第一层卷积层之后，得到的feature map的大小为(5-3+2*1)/1+1=5,所以feature map的维度为256*5*5'''
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1, stride=1)
        '''第五层池化层，卷积核大小为3*3，步距为2，前一层的feature map大小为5*5，通道数256'''
        '''第二层池化层后，feature map大小为（5-3）/2+1=2,feture map的维度为256*2*2'''
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        '''经过第一层全连接'''
        self.linear1 = nn.Linear(1024, 2048)
        self.dropout1 = nn.Dropout(0.5)
        self.linear2 = nn.Linear(2048, 2048)
        self.dropout2 = nn.Dropout(0.5)
        self.linear3 = nn.Linear(2048, 10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.pool1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = F.relu(self.conv5(out))
        out = self.pool3(out)
        out = out.reshape(-1, 256 * 2 * 2)
        out = F.relu(self.linear1(out))
        out = self.dropout1(out)
        out = F.relu(self.linear2(out))
        out = self.dropout2(out)
        out = self.linear3(out)
        return out


model = AlexNet()
model = model.to(device)
summary.summary(model, input_size=(3, 32, 32), batch_size=128, device="cuda")
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
criterion = criterion.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(model=model, epoch=total_epoch, batch_size=batch_size, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, criterion=criterion)
