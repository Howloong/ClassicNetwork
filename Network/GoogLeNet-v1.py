import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torchsummary
import torchvision
from torchvision import transforms

from utils.train_model import train_model

'''基本的卷积模块，包含一个ReLU激活函数'''


class BasicConv2d(nn.Module):
    """
    :param input_channel:输入通道数
    :param output_channel:输出通道数
    """

    def __init__(self, input_channel, output_channel, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.batchNorm = nn.BatchNorm2d(output_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.batchNorm(x)
        x = self.relu(x)

        return x


'''Inception模块'''
'''共4个分支，是并联关系，需要一起返回'''


class Inception(nn.Module):
    def __init__(self, in_channel, ch1x1, ch3x3reduce, ch3x3, ch5x5reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = BasicConv2d(in_channel, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, ch3x3reduce, kernel_size=1),
            BasicConv2d(ch3x3reduce, ch3x3, kernel_size=3, padding=1)
            # 保证输入大小等于输出大小
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, ch5x5reduce, kernel_size=1),
            BasicConv2d(ch5x5reduce, ch5x5, kernel_size=5, padding=2)
            # 保证输入大小等于输出大小
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channel, pool_proj, kernel_size=1)
            # 同样保证输入大小等于输出
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        outputs = [branch1, branch2, branch3, branch4]
        # 1代表横着拼
        return torch.cat(outputs, 1)


'''总的流程：
输入特征矩阵x -> 平均池化AvgPool2d -> 卷积BasicConv2d -> 降维 flatten -> 
dropout -> 激活Relu -> 全连接fc
'''


class InceptionAux(nn.Module):

    def __init__(self, in_channel, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = BasicConv2d(in_channel, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # aux1:N*512*4*4  aux2:N*528*4*4
        # 首先平均池化，下采样，核大小为5，步频3
        x = self.averagePool(x)
        # 卷积激活
        x = self.conv(x)
        # 拉平，三维变二维
        x = torch.flatten(x, 1)
        # 0.7的概率dropout
        x = F.dropout(x, 0.5, training=self.training)
        # 全连接得到分类的一维向量，激活
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)
        return x


class GoogLeNet(nn.Module):
    def __init__(self, num_classes=1000, aux_logits=True, init_weights=False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxPool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.conv2 = BasicConv2d(64, 64, kernel_size=1)
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)
        self.maxPool2 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxPool3 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxPool4 = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initalize_weights()

    def forward(self, x):
        # 上面是输入,下面是输出,格式为 num*C*H*W
        # N*3*224*224
        x = self.conv1(x)
        # N*64*112*112
        x = self.maxPool1(x)
        # N*64*56*56
        x = self.conv2(x)
        # N*64*56*56
        x = self.conv3(x)
        # N*192*56*56
        x = self.maxPool2(x)
        # N*192*28*28

        # N*192*28*28
        x = self.inception3a(x)
        # N*256*28*28
        x = self.inception3b(x)
        # N*480*28*28
        x = self.maxPool3(x)
        # N*480*14*14
        x = self.inception4a(x)
        # N*512*14*14
        # if self.training and self.aux_logits:
        #     aux1 = self.aux1(x)

        # N*512*14*14
        x = self.inception4b(x)
        # N*512*14*14
        x = self.inception4c(x)
        # N*512*14*14
        x = self.inception4d(x)
        # N*528*14*14

        # if self.training and self.aux_logits:
        #     aux2 = self.aux2(x)

        x = self.inception4e(x)
        # N*832*14*14
        x = self.maxPool4(x)
        # N*832*7*7
        x = self.inception5a(x)
        # N*832*7*7
        x = self.inception5b(x)
        # N*1024*7*7

        x = self.avgPool(x)
        # N*1024*1*1
        x = torch.flatten(x, 1)
        # N*1024
        x = self.dropout(x)
        x = self.fc(x)
        # N*1000
        # if self.training and self.aux_logits:
        #     return x, aux1, aux2
        return x

    def _initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # 使用何氏初始化填充权重，fan_out表示保存前向传播中权值的变化大小，使用relu。
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            #         使常数将偏置项置0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = GoogLeNet(num_classes=10).to(device)
batch_size = 2000
epoch = 40
# 载入数据集
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)
training_set = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)
train_loader = torch.utils.data.dataloader.DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False)

torchsummary.summary(model, input_size=(3, 32, 32), batch_size=batch_size, device=device.type)
# 定义损失函数：交叉熵
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# 优化函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 总batch数
train_model(model=model, epoch=epoch, batch_size=batch_size, train_loader=train_loader, test_loader=test_loader,
            optimizer=optimizer, criterion=criterion, device=device)
