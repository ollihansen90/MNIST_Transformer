import torch
import torch.nn as nn
from transformer import Attention
from torch.utils.data import Dataset
from dataset import MNIST_data
import matplotlib.pyplot as plt
from datetime import datetime as dt
from patchify import patchify

inner_dim = 2*49
projector = nn.Linear(49, inner_dim)
att = Attention(dim=inner_dim, heads=1, dim_head=98)
#img = 1
dataset = MNIST_data(labels=[1,3,0])
print(len(dataset))
img = dataset[50]

img = patchify(img)
img = projector(img)
img = att(img)
print(img.shape)


"""plt.figure()
plt.imshow(img)
plt.savefig("plots/"+str(round(dt.now().timestamp()))+".png")"""