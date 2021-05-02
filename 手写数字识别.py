# torchvision 图片 torchtext 文本
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import numpy as np
BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000


# 1准备数据
def get_loader(train=True,batch_size=BATCH_SIZE):
    # 下载数据集,准备数据
    dataset = MNIST(root="./data/", train=train, download=False, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # mean和std的形状 和通道数相同
    ]))

    # 组装dataloader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


# 2 构建模型
from torch import nn
from torch.nn.functional import relu,log_softmax,nll_loss
from torch.optim import Adam

class MinstModel(nn.Module):
    def __init__(self):
        super(MinstModel,self).__init__()
        self.fc1 = nn.Linear(1*28*28,28) # 全连接
        self.fc2 = nn.Linear(28,10) # 输出1层 10个数

    def forward(self,x):
        """

        :param x: [batch_size,1,28,28] 四维
        :return:
        """
        # 1.修改形状,变成二维的，可以进行矩阵乘
        x = x.view([-1,1*28*28])
        # 2. 进行全连接
        x = self.fc1(x)
        # 3.激活函数,形状没变化
        x = relu(x)
        out = self.fc2(x)

        # 使用 softmax计算损失，交叉熵损失
        return log_softmax(out,dim=-1)


def train(epoch):
    """
    实现训练过程
    :param epoch:
    :return:
    """
    model = MinstModel()
    optimizer = Adam(model.parameters(),lr=0.001)
    data_loader = get_loader()
    for i in range(epoch):
        for idx, (input, taeget) in enumerate(data_loader):
            optimizer.zero_grad()
            output = model(input)
            loss = nll_loss(output, taeget)  # 获取损失
            loss.backward()
            optimizer.step()  # 更新参数
            # 保持模型
            torch.save(model.state_dict(),"./model/model.pk")
            torch.save(optimizer.state_dict(), "./model/optimizer.pk")
    return model


def test(model):
    """
    评估模型
    :return:
    """
    loss_list = []
    acc_list = []
    test_dataloader = get_loader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            output = model(input)
            cur_loss = nll_loss(output,target)
            loss_list.append(cur_loss)
            # 计算准确率
            predict = output.max(dim=-1)[-1]
            cur_acc = predict.eq(target).float().mean()
            acc_list.append(cur_acc)

    print("平均准确率，平均损失",np.mean(acc_list),np.mean(loss_list))

def aa():
    pass


if __name__ == '__main__':
    model = train(3)
    test(model)











