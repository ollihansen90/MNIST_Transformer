import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime as dt

model = torch.load("models/model.pt").cpu().eval()
"""for param in model.parameters():
    print(param.shape)
dataset = MNIST_data()
dataloader = Dataloader(dataset)"""
dataset = dset.EMNIST(
    root="datasets",
    split="byclass",
    #split="balanced",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
print("targets", dataset.targets.numpy().shape)
n_data = len(dataset)
batch_size = 1000
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
n_classes = 62
print(dataset.classes)

for batch in dataloader:
    img, label = batch
    break
output = model(img)
print("output", label.shape)
print("output", output.shape)
_, idx = torch.max(output, dim=-1)
#_, real_idx = torch.max(label, dim=-1)
real_idx = label

print("Top1-Acc", torch.sum(torch.argmax(output, dim=-1)==label).item()/batch_size)

confusion = torch.zeros((n_classes,n_classes))
for i,j in zip(idx, real_idx):
    confusion[i,j] += 1

confusion = F.normalize(confusion, p=1, dim=0)

plt.figure()
plt.imshow(confusion)
#plt.gca().set_xticklabels(dataset.targets.item())
plt.savefig("plots/plot_{}_C.png".format(round(dt.now().timestamp())))