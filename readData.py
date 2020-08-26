from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np

class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_path = os.listdir(self.path)
        pass

    def __getitem__(self, indx):
        img_name = self.img_path[indx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        img_arr = np.array(img)
        label =self.label_dir
        return img_arr, label
        pass

    def __len__(self):
        return len(self.img_path)
        pass
    pass

root_dir = "dataset"
ants_label_dir = "ants"
bees_label_dir = "bees"

ants_dataset = MyData(root_dir,ants_label_dir)
bees_dataset = MyData(root_dir,bees_label_dir)

train_dataset = ants_dataset + bees_dataset
