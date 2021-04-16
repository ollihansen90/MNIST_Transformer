import torch
from dataloader import Dataloader
from dataset import MNIST_data
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime as dt

model = torch.load("models/model.pt").cpu().eval()
dataset = MNIST_data()
dataloader = Dataloader(dataset)

img, label = dataloader.getbatch(n=1000)
output = model(img)
print(output.shape)
_, idx = torch.max(output, dim=-1)
_, real_idx = torch.max(label, dim=-1)

confusion = torch.zeros((10,10))
for i,j in zip(idx, real_idx):
    confusion[i,j] += 1

confusion = F.normalize(confusion, p=1, dim=0)

plt.figure()
plt.imshow(confusion)
plt.savefig("plots/plot_{}_C.png".format(round(dt.now().timestamp())))