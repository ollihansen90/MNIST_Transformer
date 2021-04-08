# Hilfsfunktionen, sollten später noch in eine utils-Datei ausgelagert werden.
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import torch
import torch.nn.functional as F
import os

def upscale(img, ni=2, nj=2):
    '''
    Simple upscaling algorithm. 
    Input:
        img - Image
        ni - scaling factor in row dimension
        nj - scaling factor in col dimension
    Output:
        img - upscaled image
    '''
    size = img.shape
    img = img.unsqueeze(4)
    img = img.expand(size[0], size[1], size[2], size[3], ni)
    img = img.flatten(start_dim=-2, end_dim=-1)
    img = img.transpose(2,3)
    img = img.unsqueeze(4)
    img = img.expand(size[0], size[1], ni*size[3], size[2], nj)
    img = img.flatten(start_dim=-2, end_dim=-1).transpose(2,3)
    return img

def npimg2torchimg(img):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    size = img.shape
    out = torch.zeros(1,size[2], size[1], size[0]).to(device)
    out[0,:,:,:] = torch.from_numpy(np.moveaxis(img, [0,1,2],[2,1,0])).to(device)
    return out

def torchimg2npimg(img):
    size = img.shape
    out = np.zeros([size[2],size[3],size[1]])
    out = imagefyuint8(np.moveaxis(img.detach().cpu().numpy().squeeze(0), [0,1,2], [2,1,0]))
    return out

def imagefyuint8(img):
    out = img-np.min(img)
    #print(np.min(out), np.max(out))
    if not np.max(out)==0:
        out = np.uint8(255/np.max(out)*out)
    return out

def kitty(input1, input2):
    input1, input2 = imgpadding(input1, input2)
    out = torch.cat([input1, input2], 1)
    return out

def imgpadding(input1, input2):
    size1, size2 = input1.shape, input2.shape
    for i in range(2,4):
        s = size1[i]-size2[i]
        tup = (0,0,0,1) if i==2 else (0,1,0,0)
        if s>0:
            input2 = F.pad(input2, tuple([s*x for x in tup]))
        elif s<0:
            input1 = F.pad(input1, tuple([-s*x for x in tup]))
    
    return input1, input2

def clipimg(img):
    img[img>255] = 255
    img[img<0] = 0
    return img

def permlist(liste, n=None):
    if not n and not n==0:
        n = len(liste)
    N = len(liste)
    output = []
    for i in range(n):
        zv = np.random.randint(N-i)
        output.append(liste[zv])
        liste[zv] = liste[-i-1]
    return output

def tinify(img, maxval):
    while img.shape[2]*img.shape[3]>maxval:
        #print(img.shape)
        img = img[:,:,::2,::2]
    #print(img.shape)
    return img

def reshapeimg(img):
    i,j = img.shape[2], img.shape[3]
    return img[:, :, :int(i/2)*2, :int(j/2)*2]
"""def getfulllist():
    path = "DIPGAN-Bilder"
    liste1 = os.listdir(path)
    for entry in liste1:
        path1 = path+"/"+entry
        if "Epoch" in entry:
            for 
    return sorted(pathlist)"""