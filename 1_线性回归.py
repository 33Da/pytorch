import torch

learning_rate = 0.01
# 准备数据
x = torch.rand([500, 1])  # 500行1列
y_true = x * 3 + 0.8

# 通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)

# 4. 通过循环，反向传播，更新参数
# for i in range(2000):
#
#     y_predict = torch.matmul(x, w) + b  # matmul矩阵乘法
#     # 3  计算loss
#     loss = (y_true - y_predict).pow(2).mean()
#
#     if w.grad is not None:  # 计算了之前的梯度
#         w.data.zero_()  # 清空之前的梯度
#     if b.grad is not None:
#         b.data.zero_()
#
#     loss.backward()  # 反向传播
#     w.data = w.data - learning_rate * w.grad
#     b.data = b.data - learning_rate * b.grad
#     if i % 50 == 0:
#         print("w,b,loss:", w.item(), b.item(), loss.item())
# # 画图
# import matplotlib.pyplot as plt
# plt.figure(figsize=(20,8))
# # 正确的线
# plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
# y_predict = torch.matmul(x,w) + b
# # 预测的线
# plt.scatter(x.numpy().reshape(-1),y_predict.detach().numpy().reshape(-1),c="r")
# plt.show()

##########################################################################
# 用api实现
from torch import nn
from torch.optim import SGD


class Lr(nn.Module):
    def __init__(self):
        super(Lr, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入和输出的列数，特征数量

    def forward(self, x):
        """
        __call__方法实际调了forward
        """
        out = self.linear(x)
        return out


model = Lr()
optimizer = SGD(model.parameters(), 0.01)  # 获取参数,学习率
loss_fn = nn.MSELoss()  # 定义损失

for i in range(20000):
    # 得到预测值
    predict = model(x)  # 调用forward
    loss = loss_fn(predict, y_true)
    # 梯度置为0
    optimizer.zero_grad()
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    if i % 500 == 0:
        params = list(model.parameters())
        print("w,b,loss:", params[0].item(), params[1].item(), loss.item())


model.eval() # 设置为评估模式
predict = model(x)

# 画图
import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
# 正确的线
plt.scatter(x.numpy().reshape(-1),y_true.numpy().reshape(-1))
# 预测的线
plt.scatter(x.numpy().reshape(-1),predict.data.numpy(),c="r")
plt.show()