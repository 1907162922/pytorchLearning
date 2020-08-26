from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# python-> tensor 数据类型
# 通过 transforms.ToTensor 看两个问题
# 1、 transforms 如何使用（python）
# 2、 为什么需要 Tensor 数据类型


img_path = "dataset/bees/10870992_eebeeb3a12.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")

# 1、 transforms 如何使用（python）
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img) # 将img的图片转化为 tensor 类型

writer.add_image("Tensor_img", tensor_img)
writer.close()



