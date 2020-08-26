import numpy as np
import torch

# 做一个二层神经网络
# 全连接 ReLU 神经网络， 一个隐藏层， 没有 bias ，用来从 x 预测 y ，使用L2 Loss
# 用 PyTorch Tensor 来计算 前向神经网络，Loss 和反向传播
# ReLU
# h = W1X + b1
# a = max(0, h)
# y(hat) = W2a + b2

# 计算一个神经网络的步骤
# forward pass 前向传播
# loss
# backward pass 反向传播


# 64个输入，1000维输入，10维输出，H 则代表中间的 hidden layer?
N, D_in, H, D_out = 64, 1000, 100, 10
# 生成数据
x = torch.randn(N, D_in)  # 生成 64*1000 的张量
y = torch.randn(N, D_out) # 生成 64*10 的张量
w1 = torch.randn(D_in, H)
w2 = torch.randn(H, D_out)

learning_rate = 1e-6

for t in range(500):
    # Forward pass
    h = x.mm(w1)  # N * H
    h_relu = h.clamp(min=0)  # 激活函数
    y_pred = h_relu.mm(w2)  # N * D_out

    # compute loss
    loss = (y_pred - y).pow(2).sum().item() # item() 是将 tensor 类型转化为数字
    # print(t, loss)

    # Backward pass
    # Compute the gradient
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.T)
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # update weights of w1 and w2
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2
    pass

# print(y_pred - y)