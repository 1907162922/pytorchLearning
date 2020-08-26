import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

LR = 0.001
BATCH_SIZE = 50
EPOCH = 1
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(
    root='./mnist_data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)

train_loader = Data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

data = torchvision.datasets.MNIST(
    root='./mnist_data',
    train=False,
)

test_x = torch.unsqueeze(data.data, dim=1).type(torch.FloatTensor)[:2000]/255
test_y = data.targets[:2000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(   # 1*28*28
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2, # if stride = 1 ,padding = (kernel_size-1)/2 = (5-1)/2
            ),# 16*28*28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),# 16*14*14
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(   # 16*14*14
                in_channels=16,
                out_channels=32,
                kernel_size=5,
                stride=1,
                padding=2,  # if stride = 1 ,padding = (kernel_size-1)/2 = (5-1)/2
            ),# 32*14*14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),# 32*7*7
        )
        self.out = nn.Linear(32*7*7, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)  # (batch, 32, 7, 7)
        x = x.view(x.size(0), -1) # (batch, 32*7*7)
        x = self.out(x)
        return x

cnn = CNN()

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x,y) in enumerate(train_loader):
        predict = cnn(x)

        loss = loss_fn(predict,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            test_output = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            acc = sum(pred_y == test_y) / test_y.size(0)
            print("Step:",step," loss: ", loss.data.item())
test_output = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'predict number')
print(test_y[:10].numpy(), 'real number')
