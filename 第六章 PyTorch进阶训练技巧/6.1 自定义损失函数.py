'''
在实际应用中，PyTorch已经集成了很多的Loss function，但有时我们想自定义自己的损失函数
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
# 1.利用函数实现
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss

# 2.以类方式定义
# 我们如果看每一个损失函数的继承关系我们就可以发现Loss函数部分继承自_loss, 部分继承自_WeightedLoss, 而_WeightedLoss继承自_loss， _loss继承自 nn.Module
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()
        self.act = nn.Sigmoid()
    def forward(self, inputs, targets, smooth=1):
        inputs = self.act(inputs)
        # 或者写成：inputs = F.sigmoid(inputs)也可以
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice

# 使用方法
input = torch.randn([2,3])
targets = torch.randn([2, 3])
criterion = DiceLoss()
loss = criterion(input, targets)
print(loss)

# 在自定义损失函数时，涉及到数学运算时，我们最好全程使用PyTorch提供的张量计算接口，这样就不需要我们实现自动求导功能并且我们可以直接调用cuda
