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
labellist = list(range(10))
model = VisualTransformer(inner_dim=20, num_classes=len(labellist)).to(device)
#dataset = MNIST_data(labels=labellist)
#dataloader = Dataloader(dataset, labels=labellist)
dataset = dset.MNIST(
    root="data2",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
batch_size = 2048
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

lr = 1e-3
betas = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

n_epochs = 650
lossliste = torch.zeros(n_epochs).to(device)
entropieliste = torch.zeros(n_epochs).to(device)

"""for epoch in range(n_epochs):
    optimizer.zero_grad()
    img, label = dataloader.getbatch(n=batch_size)
    img = img.to(device)
    label = label.unsqueeze(-2).to(device)
    output = model(img)
    loss = torch.sum((label-output)**2)#-0.2*torch.sum(output*torch.log(output))
    #print(label-output)
    preentropy = output.detach()
    entropieliste[epoch] = -torch.sum(preentropy*torch.log(preentropy))
    loss.backward()
    optimizer.step()
    lossliste[epoch] = loss.detach()
    if epoch%200==0:
        print(epoch, loss)"""
for epoch in range(n_epochs):
    epochstart = dt.now().timestamp()
    total_loss = 0
    for batch in dataloader:
        img, labels = batch
        labelsmat = F.one_hot(labels, num_classes=10).to(device)
        output = model(img.to(device))
        loss = torch.sum((output-labelsmat)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    lossliste[epoch] = total_loss
    print("epoch", epoch, "\t| loss:", total_loss, "\t| time:", dt.now().timestamp()-epochstart, "\t| time total:", dt.now().timestamp()-starttime)

if plotstuff:
    plt.figure()
    plt.plot(lossliste.cpu(), "b.")
    plt.savefig("plots/plot_{}.png".format(round(dt.now().timestamp())))

    plt.figure()
    plt.plot(entropieliste.cpu(), "b.")
    plt.savefig("plots/plot_{}_H.png".format(round(dt.now().timestamp())))

if savemodel:
    torch.save(model, "models/model.pt")