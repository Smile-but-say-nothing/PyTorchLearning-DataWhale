# PyTorch中没有像tf那样的model.summary()函数，但有torchinfo
import torchvision.models as models
# 直接print
model = models.resnet18()
# 现单纯的print(model)，只能得出基础构件的信息，既不能显示出每一层的shape，也不能显示对应参数量的大小
print(model)

# torchinfo
from torchinfo import summary
resnet18 = models.resnet18()
summary(resnet18, (1, 3, 224, 224))