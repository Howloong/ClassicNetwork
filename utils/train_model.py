import time

import torch
import torch.utils.data.dataloader as dataloader
import torchvision
from torchvision import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_dataset(batch_size=100, ):
    print(device)
    '''定义参数'''

    '''获取数据集'''
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root='../data/', train=True, download=True,
        transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(
        root='../data/', train=False,
        transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def train_model(model, epoch, batch_size, train_loader, test_loader, optimizer,
                criterion=torch.nn.CrossEntropyLoss()
                ):
    criterion = criterion.to(device)
    model = model.to(device)
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    total_step = len(train_loader)
    time1 = time.time()
    for e in range(epoch):
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if (i + 1) % 100 == 0:
            print(f'Epoch:[{e + 1}/{epoch}],Step:[{i + 1}/{total_step}],Loss:[{loss.item() : .4f}]')
    time2 = time.time()
    print(f"time cost:{time2 - time1}s")

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predictions = torch.max(outputs.data, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print("Accuracy for class {:5s} is: {:.1f} %".format(classname,
                                                             accuracy))
