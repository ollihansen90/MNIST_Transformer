import torch
import torch.nn.functional as F
from transformer import VisualTransformer
#from dataloader import Dataloader
#from dataset import MNIST_data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import matplotlib.pyplot as plt
from datetime import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"
plotstuff = 1
savemodel = 1
starttime = dt.now().timestamp()

#labellist = [3,4,7]
labellist = list(range(62))
num_classes = 47
model = VisualTransformer(inner_dim=49*2, transformer_depth=4, num_classes=num_classes).to(device)
print(sum([params.numel() for params in model.parameters()]))
#dataset = MNIST_data(labels=labellist)
#dataloader = Dataloader(dataset, labels=labellist)
dataset = dset.EMNIST(
    root="datasets",
    split="balanced",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
n_data = len(dataset)
batch_size = 2**12
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(n_data)

lr = 1e-3
betas = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

n_epochs = 200
lossliste = torch.zeros(n_epochs).to(device)
entropieliste = torch.zeros(n_epochs).to(device)
accliste = torch.zeros(n_epochs).to(device)

for epoch in range(n_epochs):
    epochstart = dt.now().timestamp()
    total_loss = 0
    acc = 0
    for batch in dataloader:
        img, labels = batch
        img, labels = img.to(device), labels.to(device)
        #print(labels[0])
        #labelsmat = F.one_hot(labels, num_classes=10).to(device)
        output = model(img)
        #loss = torch.sum((output-labelsmat)**2)
        loss = F.cross_entropy(output, labels)
        acc += torch.sum(torch.argmax(output, dim=-1)==labels)#.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()
    
    acc = acc.item()/n_data
    accliste[epoch] = acc
    lossliste[epoch] = total_loss.item()
    plottime = dt.now().timestamp()
    print(  "epoch", epoch, 
            "\t| acc:", round(acc, 4),
            "\t| loss:", round(total_loss.item(),3), 
            "\t| time:", round(plottime-epochstart, 2), 
            "\t| time total:", round(plottime-starttime,2), "\t", dt.fromtimestamp(plottime-starttime).strftime("%H:%M:%S")
    )

if plotstuff:
    plt.figure()
    plt.plot(lossliste.cpu(), "b.")
    plt.savefig("plots/plot_{}.png".format(round(starttime)))

    plt.figure()
    plt.plot(accliste.cpu(), "b.")
    plt.savefig("plots/plot_{}_A.png".format(round(starttime)))

    plt.figure()
    plt.plot(entropieliste.cpu(), "b.")
    plt.savefig("plots/plot_{}_H.png".format(round(starttime)))

if savemodel:
    torch.save(model, "models/model_{}.pt".format(round(starttime)))
    print("Model saved, {}".format(round(starttime)))