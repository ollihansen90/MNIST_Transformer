import torch
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from datetime import datetime as dt
from utils import npimg2torchimg, torchimg2npimg

def patchify(img, n_patches=16):
    """img = torch.tensor_split(img, n_patches_x, dim=-1)
    img = torch.stack(img, dim=0).flatten(start_dim=0, end_dim=1)
    img = torch.tensor_split(img, n_patches_y, dim=-2)
    img = torch.stack(img, dim=0).flatten(start_dim=0, end_dim=1)"""
    img = torch.stack(torch.chunk(img, int(sqrt(n_patches)), dim=-1), dim=-3)
    #print(img.shape)
    img = torch.cat(torch.chunk(img, int(sqrt(n_patches)), dim=-2), dim=-3)
    img = img.flatten(start_dim=-2, end_dim=-1)
    #print("hier", img.shape)
    #img = img.flatten(start_dim=-4, end_dim=-3)
    return img.squeeze()

def buildgrid():
    n = 4
    grid = torch.tensor(list(range(n)))
    grid = grid.unsqueeze(0)
    grid = grid.expand(n,n)
    grid = grid + n*torch.tensor(list(range(n))).unsqueeze(1)
    return grid.t().flatten()

if __name__ == "__main__":
    datastack = torch.zeros([9,1,28,28])
    #print(np.expand_dims(plt.imread("data/{}/0.png".format(0)), 2).shape)
    for i in range(9):
        datastack[i,:,:,:] = npimg2torchimg(np.expand_dims(plt.imread("data/{}/0.png".format(i)), 2))
    print(datastack.shape)

    """plt.figure()
    for i in range(9):
        plt.subplot(1,9,i+1)
        plt.imshow(torchimg2npimg(datastack[i,:,:,:].unsqueeze(0)).squeeze())
    plt.show()"""

    img = datastack[3,:,:,:].unsqueeze(0)
    print(img.shape)
    img_patched = patchify(img)
    print(img_patched.shape)

    grid = buildgrid()
    plt.figure()
    for i in range(16):
        #print(i)
        plt.subplot(4,4,i+1)
        img2plot = torchimg2npimg(img_patched[grid[i],:,:,:].unsqueeze(0)).squeeze()
        #print(img2plot.shape)
        plt.imshow(img2plot)
        plt.title(i)

    plt.savefig("plots/"+str(round(dt.now().timestamp()))+".png")



