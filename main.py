import torch
from transformer import VisualTransformer
from dataloader import Dataloader
from dataset import MNIST_data
import matplotlib.pyplot as plt
from datetime import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"

#labellist = [3,4,7]
labellist = list(range(10))
model = VisualTransformer(inner_dim=49*2, num_classes=len(labellist)).to(device)
dataset = MNIST_data(labels=labellist)
dataloader = Dataloader(dataset, labels=labellist)

lr = 1e-5
betas = (0.9, 0.999)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)#.to(device)
batch_size = 32

n_epochs = 100_000
lossliste = torch.zeros(n_epochs).to(device)
entropieliste = torch.zeros(n_epochs).to(device)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    img, label = dataloader.getbatch(n=batch_size)
    img = img.to(device)
    label = label.unsqueeze(-2).to(device)
    output = model(img)
    #print("label", label.shape)
    #print("output", output.shape)
    #print(img.shape)
    loss = torch.sum((label-output)**2)#-0.2*torch.sum(output*torch.log(output))
    #print(label-output)
    preentropy = output.detach()
    entropieliste[epoch] = -torch.sum(preentropy*torch.log(preentropy))
    loss.backward()
    optimizer.step()
    lossliste[epoch] = loss.detach()
    if epoch%200==0:
        print(epoch, loss)

plt.figure()
plt.plot(lossliste.cpu(), "b.")
plt.savefig("plots/plot_{}.png".format(round(dt.now().timestamp())))

plt.figure()
plt.plot(entropieliste.cpu(), "b.")
plt.savefig("plots/plot_{}_H.png".format(round(dt.now().timestamp())))

torch.save(model, "models/model.pt")