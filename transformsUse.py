from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

writer = SummaryWriter("logs")
img = Image.open("dataset/bees/144098310_a4176fd54d.jpg")
print(img)

# 将图片转为 Tensor 类型 采用 ToTensor
trans_tensor = transforms.ToTensor()
img_tensor = trans_tensor(img)
writer.add_image("Tensor_img", img_tensor)
print(img_tensor[0][0][0]) # 原来的值

# 再转 Normalize 将图片进行变换
trans_norm = transforms.Normalize([0.8, 0.6, 0.8], [0.6, 0.5, 0.8])
# 计算公式 output[channel] = (input[channel] - mean[channel]) / std[channel]
img_norm = trans_norm(img_tensor) #这里参数为 tensor 类型
print(img_norm[0][0][0]) # 变换后新的值
writer.add_image("Normlize_img", img_norm)

# Resize
print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
img_resize = trans_tensor(img_resize)
print(img_resize)
writer.add_image("Resize", img_resize, 0)

writer.close()