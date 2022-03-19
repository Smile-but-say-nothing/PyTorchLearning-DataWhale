from __future__ import print_function
import torch
'''
Tensor is actually a n-dimension array:
    d = 0, a scalar or a pure number
    d = 1, a vector
    d = 2, a matrix
    d = 3, time series data, RGB image...
    d = 4, a batch images: (sample_size, width, height, channel)
    d = ...
'''
# Create a tensor with 4 rows and 3 columns
x = torch.rand(4, 3)  # torch.rand([4, 3])
print(x)
x = torch.zeros([4, 3], dtype=torch.long)
print(x)
# A tensor with assigned value
x = torch.Tensor([3, 2])
print(x)
x = torch.Tensor(x)
print(x)
# Create a new tensor with tensor before
y = torch.rand_like(x, dtype=torch.float)
print(y)
# Use .size() or .shape to get the shape of the tensor
y = torch.rand([18, 224, 224, 3])
print(f'y.size: {y.size()}, y.shape: {y.shape}')
# Add method by element-wise
x = torch.rand([12, 224, 224, 3])
y = torch.ones_like(x, dtype=torch.float)
z = torch.add(x, y)
print(x, y, z)
print(f'z.shape: {z.shape}')
# Index
x = torch.rand([4,3], dtype=torch.float)
y = x[0, :3]
print(x, y)
'''
Use tensor.view to change the shape of the tensor, but notice that this way do not open up a new region in memory, if 
want to do so, use clone + view or reshape (not recommend, because reshape() don't guarantee return a copy of source tensor)
'''
x = torch.rand([4, 4], dtype=torch.float)
print(f"Old: {x}")
x = x.view(16)
print(f"New: {x}")
y = x
# If we change x, y will be also changed because it is a reference of x
x[3] = 0
print(x, y)
x = torch.rand([4, 4], dtype=torch.float)
# Clone + view
x = x.view(16)
y = x.clone()
x[3] = 0
y = y.view(2, 8)
print(x, y)
# Use .item to get the value in tensor(scalar)
y = torch.rand(1)
print(y.item())
# Broadcasting
x = torch.arange(1, 3).view(1, 2)  # tensor([[1, 2]])
'''tensor([[1, 2],
            [1, 2],
            [1, 2]])'''
print(x)
y = torch.arange(1, 4).view(3, 1)
'''tensor([[1, 1],
        [2, 2],
        [3, 3]])'''
print(y)
print(x + y)  # element-wise add
'''tensor([[2, 3],
        [3, 4],
        [4, 5]])'''
print(torch.add(x, y))  # same result
x = torch.arange(1, 13).view(2, 3, 2)
print(x)
'''
1.所有数组向最长的数组看齐，不足维度为1
2.输出数组的shape是所有输入数组shape各个axis上最大值
3.如果输入数组的某个axis和输出数组对应axis的长度相同或为1，可以进行计算，否则出错
4.当输入数组的某个axis长度为1时，在此axis上的计算都是复制此axis的第1组值
'''
y = torch.arange(1, 4).view(3, 1)  # 3x1 -> 1x3x1 -> 2x3x2
print(y)
print(x + y)