"""import torch
import torch.nn as nn
from transformer import VisualTransformer
from dataset import MNIST_data
import matplotlib.pyplot as plt
from datetime import datetime as dt
from patchify import patchify
from dataloader import Dataloader

inner_dim = 2*49
projector = nn.Linear(49, inner_dim)
#att = Attention(dim=inner_dim, heads=8, dim_head=98)
#transf = Transformer(dim=inner_dim)
vit = VisualTransformer()
dataset = MNIST_data(labels=[1,3,0])
dataloader = Dataloader(dataset, num_classes=dataset.num_classes)

batch, label = dataloader.getbatch(n=4)
print(batch.shape)
print(label)

#img = patchify(img)
#img = projector(img)
#img = att(img)
#img = transf(img)
img = vit(batch)
print(img)
#print(label)

"""
"""plt.figure()
plt.imshow(img)
plt.savefig("plots/"+str(round(dt.now().timestamp()))+".png")"""

import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)
