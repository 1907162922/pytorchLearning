import torch
import torch.utils.data as Data

BATCH_SIZE = 5

x = torch.linspace(1,10,10)
y = torch.linspace(10,1,10)

torch_dataset = Data.TensorDataset(x,y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
) # loader 作用是进行一小批一小批的训练
for epoch in range(3):
    for step, (batch_x, batch_y) in enumerate(loader):
        print('Epoch: ',epoch,' Step: ',step,
              'Batch_x: ',batch_x.numpy(),' Batch_y: ',batch_y.numpy())
