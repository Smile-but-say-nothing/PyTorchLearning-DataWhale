# 如何修改已经开源的模型？
import torchvision.models as models
net = models.resnet18()
print(net)
# (fc): Linear(in_features=512, out_features=1000, bias=True) 起初针对的是ImageNet
# 如果想修改为十分类问题，且增加一层全连接
from collections import OrderedDict
import torch.nn as nn
classifier =  nn.Sequential(
    OrderedDict([
        ('fc1', nn.Linear(512, 128)),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(128, 10)),
        ('output', nn.Softmax(dim=1))
    ])
)
net.fc = classifier
# Sequential+OrderedDict的模型定义方式，实现十分类问题改装模型
print(net)

# 添加外部输入
import torch
"""
有时候在模型训练中，除了已有模型的输入之外，还需要输入额外的信息。
比如在CNN网络中，我们除了输入图像，还需要同时输入图像对应的其他信息，这时候就需要在已有的CNN网络中添加额外的输入变量。
基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改。
多模态
"""
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        # 在倒数第二层添加新的输入和层
        self.net = net
        # 新的层
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(3048, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x = self.net(x)
        # add_variable.unsqueeze(1)是为了和net输出的tensor保持维度一致，常用于add_variable是单一数值 (scalar) 的情况，
        # 此时add_variable的维度是 (batch_size, )，需要在第二维补充维数1，从而可以和tensor进行torch.cat操作。
        # x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)), 1)
        x = torch.cat((self.dropout(self.relu(x)), add_variable), 1)
        x = self.fc_add(x)
        x = self.output(x)
        return x


net = models.resnet50()
model = Model(net)
inputs = torch.rand([1, 3, 224, 224])
add_var = torch.rand([1, 2048])
ouputs = model(inputs, add_var)
print(ouputs)

# 添加额外输出，在forward中改return
class Model(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)

    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        # return more
        return x10, x1000

net = models.resnet50()
model = Model(net)
inputs = torch.rand([1, 3, 224, 224])
add_var = torch.rand([1, 2048])
ouputs = model(inputs, add_var)
print(ouputs)