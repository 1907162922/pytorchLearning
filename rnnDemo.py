import torch
from torch import nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28   # rnn time step / image height
INPUT_SIZE = 28  # rnn input size / image width
LR = 0.01
DOWNLOAD_MNIST = False

train_data = dsets.MNIST(
    root='./mnist_data',
    train=True,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
train_loader = torch.utils.data.DataLoader(
    dataset=train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_data = dsets.MNIST(
    root='./mnist_data',
    train=False,
    transform=transforms.ToTensor(),
    download=DOWNLOAD_MNIST,
)
test_x = test_data.test_data.type(torch.FloatTensor)[:2000]/255.
test_y = test_data.targets[:2000]

class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64,
            num_layers=1,
            batch_first=True,
        )
        self.out = nn.Linear(64, 10)

    def forward(self, x):
        r_out, (h_n, h_c) = self.rnn(x, None)  # x (batch, time_step, input_size)
        out = self.out(r_out[:, -1,:])  # (batch, time_step, input_size) -1 表示倒数第一个
        return out

rnn = RNN()
rnn = rnn.cuda()
test_x, test_y = test_x.cuda(), test_y.cuda()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        x = x.view(-1,28,28).cuda()
        y = y.cuda()
        output = rnn(x)
        loss = loss_fn(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step+1) % 100 == 0:
            test_output = rnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.squeeze()
            acc = 1.0*sum(pred_y == test_y) / test_y.size(0)
            print('Step: ',step+1,' loss: ', loss.data.item(), ' acc: %.4f'%acc)

test_out = rnn(test_x[:10])
pred_y = torch.max(test_out, 1)[1].data.squeeze()
print(pred_y.cpu().numpy())
print(test_y[:10].cpu().numpy())