import torch
from transformer import VisualTransformer
from dataloader import Dataloader
from dataset import MNIST_data
import matplotlib.pyplot as plt
from datetime import datetime as dt

device = "cuda" if torch.cuda.is_available() else "cpu"

labellist = [3,4,7]
#labellist = list(range(10))
model = VisualTransformer(inner_dim=49*2, num_classes=len(labellist)).to(device)
dataset = MNIST_data(labels=labellist)
dataloader = Dataloader(dataset, num_classes=len(labellist))

lr = 1e-3
betas = (0.95, 0.99)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas).to(device)
batch_size = 8

n_epochs = 100
lossliste = torch.zeros(n_epochs).to(device)

for epoch in range(n_epochs):
    optimizer.zero_grad()
    img, label = dataloader.getbatch(n=batch_size)
    img = img.to(device)
    label = label.to(device)
    #print(img.shape)
    loss = torch.sum((label-model(img))**2)
    loss.backward()
    optimizer.step()
    lossliste[epoch] = loss.detach()
    if epoch%200==0:
        print(epoch, loss)

plt.figure()
plt.plot(lossliste.cpu())
plt.savefig("plots/plot_{}.png".format(dt.now().timestamp()))