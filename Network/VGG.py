import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.dataloader as dataloader
import torchvision.datasets
from torchvision.transforms import transforms

from utils.train_model import train_model


class VGG(nn.Module):
    def __init__(self, arch, num_classes=1000):
        super(VGG, self).__init__()
        self.in_channel = 3
        self.conv3_64 = self.__make_layer(64, arch[0])
        self.conv3_128 = self.__make_layer(128, arch[1])
        self.conv_256 = self.__make_layer(256, arch[2])
        self.conv_512a = self.__make_layer(512, arch[3])
        self.conv_512b = self.__make_layer(512, arch[4])
        self.fc1 = nn.Linear(7 * 7 * 512, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)

    def __make_layer(self, out_channels, num):
        layers = []
        for i in range(num):
            layers.append(nn.Conv2d(self.in_channel, out_channels, kernel_size=(3, 3), stride=1, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            self.in_channel = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv3_64(x)
        out = F.max_pool2d(out, 2)
        out = self.conv3_128(out)
        out = F.max_pool2d(out, 2)
        out = self.conv_256(out)
        out = F.max_pool2d(out, 2)
        out = self.conv_512a(out)
        out = F.max_pool2d(out, 2)
        out = self.conv_512b(out)
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = F.relu(out)
        return F.softmax(out, dim=1)


def VGG_11():
    return VGG([1, 1, 2, 2, 2])


def VGG_13():
    return VGG([2, 2, 2, 2, 2])


def VGG_16():
    return VGG([2, 2, 3, 3, 3])


def VGG_19():
    return VGG([2, 2, 4, 4, 4])


device = "cuda:0" if torch.cuda.is_available() else "cpu"
vgg_11 = VGG_11().to(device)
vgg_13 = VGG_13().to(device)
vgg_16 = VGG_16().to(device)
vgg_19 = VGG_19().to(device)
# for m in [vgg_11, vgg_13, vgg_16, vgg_19]:
#     summary(m, input_size=(3, 32, 32), device=device)
#     print('***********************************')

batch_size = 50
epoch = 10
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
)
training_set = torchvision.datasets.CIFAR10(
    root='../data', train=True, download=True, transform=transform
)
test_set = torchvision.datasets.CIFAR10(
    root='../data', train=False, download=True, transform=transform
)
train_loader = torch.utils.data.dataloader.DataLoader(training_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False)
optimizer = torch.optim.Adam(vgg_11.parameters(), lr=0.03)
criterion = nn.CrossEntropyLoss().to(device)
train_model(vgg_11, epoch, batch_size, train_loader, test_loader, optimizer, criterion, device)
