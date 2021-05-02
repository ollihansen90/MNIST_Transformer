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
start_epoch = 0

#labellist = [3,4,7]
labellist = list(range(62))
num_classes = 62
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

#lr = 1e-3
betas = (0.9, 0.999)
#optimizer = torch.optim.SGD(model.parameters(), lr=lr)#, betas=betas)

n_epochs = 200

params = [5e-5, 1e-4, 3e-4, 5e-4, 1e-3]
warmuplr = 1e-5
jumplist = [0, 5, n_epochs+1]
lossliste = torch.zeros(len(params), n_epochs).to(device)
accliste_train = torch.zeros(len(params), n_epochs).to(device)
accliste_test = torch.zeros(len(params), n_epochs).to(device)
for param_idx, p in enumerate(params):
    jumpidx = 0
    starttime = dt.now().timestamp()
    model = VisualTransformer(
                        inner_dim=2*num_classes, 
                        transformer_depth=3, # Größe des Stapels an Transformern (werden nacheinander durchiteriert)
                        attn_heads=3, # Anzahl Attention Heads
                        dim_head=62, # eigene Dimension für Attention
                        mlp_dim=128, # Dimension des MLPs im Transformer
                        transformer_dropout=0.1, # Dropout des MLP im Transformer
                        num_classes=num_classes # Anzahl Klassen
                    ).to(device)
    #model = VisualTransformer(inner_dim=p, transformer_depth=1, dim_head=49, attn_heads=3, mlp_dim=49, num_classes=num_classes).to(device)
    #model = torch.load("models/"+modelnames[param_idx]+".pt")
    #model = torch.load("models/model.pt")
    print(sum([params.numel() for params in model.parameters()]))
    print("Lernrate", p)
    lr_list = [warmuplr, p]
    optimizer = Lamb(model.parameters(), lr=1e-3, betas=betas, weight_decay=0.1)
    
    with open("where.txt", "a+") as file:
        file.write("--- Lernrate "+str(p)+", "+ str(round(starttime)) + 70*"-"+"\n")
    for epoch in range(start_epoch, n_epochs+start_epoch):
        if epoch==jumplist[jumpidx]:
            for g in optimizer.param_groups:
                g["lr"]= lr_list[jumpidx]
            #optimizer = Lamb(model.parameters(), lr=lr_list[jumpidx], betas=betas, weight_decay=0.1)
            #optimizer = torch.optim.AdamW(model.parameters(), lr=lr_list[jumpidx], betas=betas, weight_decay=0.1)
            jumpidx += 1
        epochstart = dt.now().timestamp()
        total_loss = 0
        acc_train = 0
        acc_test = 0

        # Training
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
        accliste_train[param_idx, epoch-start_epoch] = acc_train
        acc_test = acc_test.item()/n_data_test
        accliste_test[param_idx, epoch-start_epoch] = acc_test
        lossliste[param_idx, epoch-start_epoch] = total_loss.item()
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
        
        if epoch%10==0 and savemodel:
            torch.save(model, "models/model.pt")

    if savemodel:
        torch.save(model, "models/model_{}.pt".format(round(starttime)))
        torch.save(model, "models/model.pt")
        print("Model saved, {}".format(round(starttime)))

if plotstuff:
    plt.figure()
    for i in range(len(params)):
        plt.plot(lossliste[i,:].cpu())
    plt.legend(params)
    plt.savefig("plots/plot_{}.png".format(round(starttime)))

    plt.figure()
    for i in range(len(params)):
        plt.plot(accliste_train[i,:].cpu())
        plt.plot(accliste_test[i,:].cpu())
    paramlist = list(["train_"+str(param), "test_"+str(param)] for param in params)
    plt.legend([x for y in paramlist for x in y])
    plt.savefig("plots/plot_{}_A.png".format(round(starttime)))

torch.save(accliste_test, "auswertungen/accliste_test_{}.pt".format(round(starttime)))
torch.save(accliste_train, "auswertungen/accliste_train_{}.pt".format(round(starttime)))
torch.save(lossliste, "auswertungen/lossliste_{}.pt".format(round(starttime)))