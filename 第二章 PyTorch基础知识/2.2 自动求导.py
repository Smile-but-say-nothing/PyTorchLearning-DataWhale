import torch
'''
torch.Tensor 是autograd的核心类。如果设置它的属性 .requires_grad 为 True，那么它将会追踪对于该张量的所有操作。
当完成计算后可以通过调用 .backward()，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到.grad属性。
注意：在 y.backward() 时，如果 y 是标量，则不需要为 backward() 传入任何参数；否则，需要传入一个与 y 同形的Tensor。

要阻止一个张量被跟踪历史，可以调用.detach()方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。
为了防止跟踪历史记录(和使用内存），可以将代码块包装在 with torch.no_grad(): 中。
在评估模型时特别有用，因为模型可能具有 requires_grad = True 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。

还有一个类对于autograd的实现非常重要：Function。Tensor 和 Function 互相连接生成了一个无环图 (acyclic graph)，它编码了完整的计算历史。
每个张量都有一个.grad_fn属性，该属性引用了创建 Tensor 自身的Function(除非这个张量是用户手动创建的，即这个张量的grad_fn是 None )。

https://zhuanlan.zhihu.com/p/83172023
'''
# .requires_grad_(...) 原地改变了现有张量的requires_grad标志。如果没有指定的话，默认输入的这个标志是 False
a = torch.randn(2, 2)  # 缺失情况下默认 requires_grad = False
a = ((a * 3) / (a - 1))
print(a.requires_grad)  # False
a.requires_grad_(True)  # 如果没有这句，b.grad_fn就是None了，也就是requires_grad可以显式的让手动创建的tensor记录梯度
print(a.requires_grad)  # True
b = (a * a).sum()
# 并不是每个requires_grad()设为True的值都会在backward的时候得到相应的grad，它还必须为leaf，b不是leaf tensor
b.backward()
print(b.grad_fn, b.grad, a.grad)  # <SumBackward0 object at 0x0000027314617400> None tensor([[3.1200, 3.6594],[1.0345, 2.2644]])

import numpy as np
x = torch.tensor(np.array([[0.3854, 0.9415], [0.8690, 0.1869]]))
# x = torch.randn([2, 3], requires_grad=True)
print('x', x)
x.requires_grad_(True)
y = x ** 2
print('y', y)
# y是x经过function的结果，所以有grad_fn
# print(y.grad_fn)  # grad_fn=<PowBackward0>，这个时候y是root tensor，x是leaf tensor有grad属性
# 有什么样的function就有相应的grad_fn
# 谁的.grad就是对谁求导的结果，多维tensor就是对每一个element求导的结果
z = y * y * 3
print('z', z)
out = z.mean()
print('out', out)
out.backward()  # 输出导数 d(out)/dx
print('x', x, '\nx.grad', x.grad)
"""
x.grad其实就是out对x的每一个element求导，以第1项x1为例，d(out)/(dx1) = d(out)/d(z) * d(z)/d(x1) = 0.25*()
"""
