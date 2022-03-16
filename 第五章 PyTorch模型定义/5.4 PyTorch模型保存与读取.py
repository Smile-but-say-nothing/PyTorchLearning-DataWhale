# PyTorch存储模型主要采用pkl，pt，pth三种格式。就使用层面来说没有区别。
""" 模型存储内容 """
"""
一个PyTorch模型主要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）。
存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。
https://github.com/datawhalechina/thorough-pytorch/blob/main/%E7%AC%AC%E4%BA%94%E7%AB%A0%20PyTorch%E6%A8%A1%E5%9E%8B%E5%AE%9A%E4%B9%89/5.4%20PyTorh%E6%A8%A1%E5%9E%8B%E4%BF%9D%E5%AD%98%E4%B8%8E%E8%AF%BB%E5%8F%96.md
"""
from torchvision import models
import torch
model = models.resnet18(pretrained=True)

save_dir = './resnet18'
# 保存整个模型，对于PyTorch而言，pt, pth和pkl三种数据格式均支持模型权重和整个模型的存储，因此使用上没有差别。
torch.save(model, save_dir + '_model')
# 保存模型权重
torch.save(model.state_dict, save_dir + '_state_dict')

# 单卡保存+单卡加载，其余情况暂时用不到
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
model.cuda()

# 保存+读取整个模型
torch.save(model, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   # 注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_dict  # 这是关键一步，其实也就是确定state_dict
loaded_model.cuda()
