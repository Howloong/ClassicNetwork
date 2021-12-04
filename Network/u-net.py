import glob
import os.path
import random

import cv2
import torch.nn as nn
import torch.nn.functional
from torch import optim
from torch.nn import init
from torch.utils.data import Dataset


class double_conv(nn.Module):

    def __init__(self, input_channel, output_channel, padding=0):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channel, out_channels=output_channel, padding=1, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(num_features=output_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=output_channel, out_channels=output_channel, padding=1, kernel_size=(3, 3), stride=1),
            # nn.BatchNorm2d(num_features=output_channel),
            nn.ReLU(inplace=True),
        )

        # self.conv.apply(self.init_weight)

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def init_weight(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


class input_conv(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(input_conv, self).__init__()
        self.conv = double_conv(input_channel=in_channel, output_channel=out_channel, padding=0)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(down, self).__init__()
        self.conv = input_conv(in_channel, out_channel)
        self.max_pool = nn.MaxPool2d(kernel_size=2, )

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x


class up(nn.Module):

    def __init__(self, in_channel, out_channel):
        super(up, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel // 2, padding=0, kernel_size=3),
            nn.ReLU(inplace=True)
        )
        self.conv = double_conv(in_channel, out_channel)
        self.up.apply(self.init_weight)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

    @staticmethod
    def init_weight(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal_(m.weight)
            init.constant_(m.bias, 0)


class uNet(nn.Module):
    def __init__(self, in_channel, in_classes):
        super(uNet, self).__init__()
        self.in_classes = in_classes
        self.pred_y = None
        self.conv1 = input_conv(in_channel, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)
        self.outConv = nn.Conv2d(64, in_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outConv(x)
        return x
        # self.pred_y = nn.functional.sigmoid(x)


class ISBILoader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.img_path = glob.glob(os.path.join(data_path, "image/*.png"))
        self.label_path = glob.glob(os.path.join(data_path, "label/*.png"))

    def augment(self, image, flipCode):
        return cv2.flip(image, flipCode)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        image_path = self.img_path[idx]
        label_path = self.label_path[idx]
        image = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
        label = cv2.imread(label_path, cv2.COLOR_BGR2GRAY)
        image = image.reshape(1, image.shape[0], image.shape[1])
        label = label.reshape(1, label.shape[0], label.shape[1])

        if label.max() > 1:
            label = label / 255
        flipCode = random.choice([-1, 0, 1, 2])
        if flipCode != 2:
            image = self.augment(image, flipCode)
            label = self.augment(label, flipCode)
        return image, label


def train_net(net, device, data_path, epochs=40, batch_size=1, lr=0.00001):
    isbi_dataset = ISBILoader('../data/ISBI/train')
    train_loader = torch.utils.data.DataLoader(dataset=isbi_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    criterion = nn.BCEWithLogitsLoss()
    best_loss = float('inf')
    for epoch in range(epochs):
        net.train()
        length = len(train_loader)
        for idx, (image, label) in enumerate(train_loader):
            optimizer.zero_grad()
            image = image.to(device, dtype=torch.float32)
            label = label.to(device, dtype=torch.float32)
            pred = net(image)
            loss = criterion(pred, label)
            print('%2d/%2d,Loss(train):%.6f' % (idx + 1, length, loss.item()))

            if loss < best_loss:
                best_loss = loss
                torch.save(net.state_dict(), 'best_model.pth')
            loss.backward()
            optimizer.step()


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() == 'cuda' else 'cpu'
    net = uNet(in_channel=1, in_classes=1)
    net.to(device)
    data_path = '../data/ISBI/train'
    train_net(net, device, data_path)
