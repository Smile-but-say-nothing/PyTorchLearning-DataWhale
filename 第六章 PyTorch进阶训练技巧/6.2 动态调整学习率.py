# lr_scheduler， 学习率调度
# lr_scheduler.LambdaLR
# lr_scheduler.MultiplicativeLR
# lr_scheduler.StepLR
# lr_scheduler.MultiStepLR
# lr_scheduler.ExponentialLR
# lr_scheduler.CosineAnnealingLR
# lr_scheduler.ReduceLROnPlateau
# lr_scheduler.CyclicLR
# lr_scheduler.OneCycleLR
# lr_scheduler.CosineAnnealingWarmRestarts
import torch
# 选择一种优化器
optimizer = torch.optim.Adam(...)
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.ExponentialLR
# scheduler2 = torch.optim.lr_scheduler....
# ...
# schedulern = torch.optim.lr_scheduler....
# 进行训练
# for epoch in range(100):
#     train(...)
#     validate(...)
#     optimizer.step()
#     # 需要在优化器参数更新之后再动态调整学习率
# 	    scheduler1.step()
# 	    ...
#     schedulern.step()
# 自定义scheduler
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr