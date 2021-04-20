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
start_epoch = 0
starttime = dt.now().timestamp()

#labellist = [3,4,7]
labellist = list(range(62))
num_classes = 62
if start_epoch:
    model = torch.load("models/model.pt")
else:
    model = VisualTransformer(inner_dim=num_classes*2, transformer_depth=1, num_classes=num_classes).to(device)
print(sum([params.numel() for params in model.parameters()]))
#dataset = MNIST_data(labels=labellist)
#dataloader = Dataloader(dataset, labels=labellist)
dataset = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)#.to(device)
n_data = len(dataset)
batch_size = 2**11+2**10
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)#.to(device)
print(n_data)

lr = 1e-3
betas = (0.9, 0.999)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=betas)

n_epochs = 20
lossliste = torch.zeros(n_epochs).to(device)
entropieliste = torch.zeros(n_epochs).to(device)
accliste = torch.zeros(n_epochs).to(device)

for epoch in range(start_epoch, n_epochs+start_epoch):
    epochstart = dt.now().timestamp()
    total_loss = 0
    acc = 0
    for img, labels in dataloader:
        #img, labels = batch
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
    accliste[epoch-start_epoch] = acc
    lossliste[epoch-start_epoch] = total_loss.item()
    plottime = dt.now().timestamp()
    line = "epoch {}\t| acc: {}\t| loss: {}\t| time: {} \t| time total: {}\t{}".format(
                epoch, 
                            round(acc, 4), 
                                        round(total_loss.item(),3), 
                                                    round(plottime-epochstart, 2), 
                                                                        round(plottime-starttime,2), 
                                                                            dt.fromtimestamp(plottime-starttime).strftime("%H:%M:%S")
                                                                                        )

    print(line)
    with open("where.txt", "a+") as file:
        file.write(line+"\n")
    
    if epoch%10==0:
        torch.save(model, "models/model.pt")

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
    torch.save(model, "models/model.pt")
    print("Model saved, {}".format(round(starttime)))