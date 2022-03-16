import torch.nn as nn
# nn.Module是PyTorch里的模型构造类，是所有神经网络模块的基类
# 我们可以通过Sequential，ModuleList和ModuleDict三种方式定义PyTorch模型。
'''
1.Sequential
该种方式的优点在于简单，快速，适合快速验证一个idea，模型的forward其实已经按照layer的顺序隐式的定义好了，但是劣势在于缺乏灵活性，一些复杂的网络结构
比如多输入，复用输出等等操作都很不方便实现，个人感觉该种方式适合简单的CNN实现
'''
net = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)
print(net)
'''
Sequential(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
'''
# 也可以通过OrderDict传入，但看起来比较复杂，好处就是可以自定义layer的name
import collections
net2 = nn.Sequential(collections.OrderedDict([
    ('fc1', nn.Linear(784, 256)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(256, 10))
]))
print(net2)
'''
Sequential(
  (fc1): Linear(in_features=784, out_features=256, bias=True)
  (relu1): ReLU()
  (fc2): Linear(in_features=256, out_features=10, bias=True)
)
'''
'''
2.ModuleList
注意该种方式本质是将各种layer存放到一起list，append的顺序并不代表网络中实际各layer的顺序,需要通过指定forward顺序
'''
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10))
net.extend([nn.ReLU(), nn.Linear(10, 5)])
print(net)
print(net[1:3])  # 可以利用index得到某一层的layer
'''
ModuleList(
  (0): Linear(in_features=784, out_features=256, bias=True)
  (1): ReLU()
  (2): Linear(in_features=256, out_features=10, bias=True)
)
'''

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.modulelist = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])

    def forward(self, x):
        for layer in self.modulelist:
            x = layer(x)
        return x
net = MyModel()
print(net)
'''
MyModel(
  (modulelist): ModuleList(
    (0): Linear(in_features=784, out_features=256, bias=True)
    (1): ReLU()
  )
)
'''
'''
3.ModuleDict
ModuleDict和ModuleList的作用类似，只是ModuleDict能够更方便地为神经网络的层添加名称。
'''
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10)  # 添加
print(net['linear'])  # 访问
print(net.output)
print(net)
'''
Linear(in_features=784, out_features=256, bias=True)
Linear(in_features=256, out_features=10, bias=True)
ModuleDict(
  (linear): Linear(in_features=784, out_features=256, bias=True)
  (act): ReLU()
  (output): Linear(in_features=256, out_features=10, bias=True)
)
'''