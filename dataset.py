import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class MNIST_data(Dataset):
    def __init__(self, main_dir="data", labels=list(range(10))):
        super(MNIST_data, self).__init__()
        self.main_dir = main_dir
        self.all_folders = [main_dir+"/{}/".format(i) for i in range(10)]
        self.all_imgs = []
        for i in labels:
            self.all_imgs += map(lambda bla: self.all_folders[i]+bla, os.listdir(self.all_folders[i]))
        self.num_classes = len(labels)
    
    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        imgloc = self.all_imgs[idx]
        label = int(imgloc.split("/")[-2])
        img = Image.open(imgloc).convert("L") # L ist ein Concept Mode: https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        img = transforms.ToTensor()(img)
        return {"image": img, "label": label}
        #img = torch.tensor(list(Image.open(imgloc)), dtype=torch.float)
        #return img
    