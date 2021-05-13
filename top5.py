import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
#import torch.nn.functional as F
from datetime import datetime as dt
from os import remove
from math import log10

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

print("Top1-Acc", torch.sum(torch.argmax(output, dim=-1)==label).item()/batch_size)
remove("top5.txt")
with open("top5.txt", "a+") as file:
    top5 = 0
    for test, out in zip(label, output):
        vals_top5, idx_top5 = torch.topk(out, 5)
        file.write(str(dataset.classes[test])+" "+ str([dataset.classes[entry] for entry in idx_top5]) + " " + str([round(log10(val.item()),2) for val in vals_top5]) + " " + str(test in idx_top5) + "\n")
        top5 += (test in idx_top5)

print("Top5-Acc", top5/batch_size)
