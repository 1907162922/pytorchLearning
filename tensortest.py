from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

writer = SummaryWriter("logs")
image_path = "dataset/ants/10308379_1b6c72e180.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')


writer.close()