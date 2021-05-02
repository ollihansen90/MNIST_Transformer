import torch
import torch.nn.functional as F
from transformer import VisualTransformer
#from dataloader import Dataloader
#from dataset import MNIST_data

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from lamb import Lamb

import matplotlib.pyplot as plt
from datetime import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"
plotstuff = 1
savemodel = 1
start_epoch = 255
starttime = dt.now().timestamp()

#labellist = [3,4,7]
labellist = list(range(62))
num_classes = 62
if start_epoch:
    model = torch.load("models/model.pt").to(device)
else:
    model = VisualTransformer(inner_dim=49, transformer_depth=1, dim_head=49, attn_heads=3, mlp_dim=49, num_classes=num_classes).to(device)
print(sum([params.numel() for params in model.parameters()]))
#dataset = MNIST_data(labels=labellist)
#dataloader = Dataloader(dataset, labels=labellist)
dataset_train = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=True,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
dataset_test = dset.EMNIST(
    root="datasets",
    split="byclass",
    train=False,
    transform=transforms.Compose([transforms.ToTensor()]),
    download=True
)
n_data_train = len(dataset_train)
n_data_test = len(dataset_test)
batch_size = 2**10
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=4)
dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=4)
print(n_data_train)
print(n_data_test)

lr = 3e-4
betas = (0.9, 0.999)
optimizer = Lamb(model.parameters(), lr=lr, betas=betas, weight_decay=0.1)
#optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=betas)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=betas)

n_epochs = 1000
lossliste = torch.zeros(n_epochs).to(device)
accliste_train = torch.zeros(n_epochs).to(device)
accliste_test = torch.zeros(n_epochs).to(device)

param_idx = 0
print("gogogo")
for epoch in range(start_epoch, n_epochs+start_epoch):
    epochstart = dt.now().timestamp()
    total_loss = 0
    acc_train = 0
    acc_test = 0

    model.train()
    for img, labels in dataloader_train:
        #img, labels = batch
        img, labels = img.to(device), labels.to(device)
        #print(labels[0])
        #labelsmat = F.one_hot(labels, num_classes=10).to(device)
        output = model(img)
        #loss = torch.sum((output-labelsmat)**2)
        loss = F.cross_entropy(output, labels)
        acc_train += torch.sum(torch.argmax(output, dim=-1)==labels)#.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach()

    # Testing
    model.eval()
    for img, labels in dataloader_test:
        #img, labels = batch
        img, labels = img.to(device), labels.to(device)
        #print(labels[0])
        #labelsmat = F.one_hot(labels, num_classes=10).to(device)
        output = model(img)
        acc_test += torch.sum(torch.argmax(output, dim=-1)==labels)

    acc_train = acc_train.item()/n_data_train
    accliste_train[epoch-start_epoch] = acc_train
    acc_test = acc_test.item()/n_data_test
    accliste_test[epoch-start_epoch] = acc_test
    lossliste[epoch-start_epoch] = total_loss.item()
    plottime = dt.now().timestamp()
    line = "{}. epoch {}\t| acc: {}, {}\t|  loss: {}\t| time: {} \t| time total: {}\t{}".format(
                str(param_idx+1).rjust(3),
                        str(epoch).rjust(3), 
                                str(round(acc_train, 4)).rjust(6),
                                        str(round(acc_test, 4)).rjust(6), 
                                                    str(round(total_loss.item(),3)).rjust(7), 
                                                                str(round(plottime-epochstart, 2)).rjust(5), 
                                                                                    str(round(plottime-starttime,2)).rjust(8), 
                                                                                        str(dt.fromtimestamp(plottime-starttime).strftime("%H:%M:%S")).rjust(8)
                                                                                            )

    print(line)
    with open("where.txt", "a+") as file:
        file.write(line+"\n")
    
    if epoch%10==0:
        torch.save(model, "models/model.pt")

if plotstuff:
    plt.figure()
    plt.plot(lossliste.cpu())
    plt.savefig("plots/plot_{}.png".format(round(starttime)))

    plt.figure()
    plt.plot(accliste_train.cpu())
    plt.plot(accliste_test.cpu())
    plt.legend(["Training", "Test"])
    plt.savefig("plots/plot_{}_A.png".format(round(starttime)))

torch.save(accliste_test, "auswertungen/accliste_test_{}.pt".format(round(starttime)))
torch.save(accliste_train, "auswertungen/accliste_train_{}.pt".format(round(starttime)))
torch.save(lossliste, "auswertungen/lossliste_{}.pt".format(round(starttime)))
if savemodel:
    torch.save(model, "models/model_{}.pt".format(round(starttime)))
    torch.save(model, "models/model.pt")
    print("Model saved, {}".format(round(starttime)))