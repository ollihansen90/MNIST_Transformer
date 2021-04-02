import os
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image

datafile = "raw/train.csv"
N_max = 1e10
counterarray = np.zeros(10)

with open(datafile, "r") as file:
    i = 0
    for line in file.readlines():
        if i==0:
            i = i+1
            continue
        img = np.array(line[:-1].split(","), dtype=np.uint8)
        im = Image.fromarray(img[1:].reshape([28,28]))
        im.save("data/{}/{}.png".format(img[0], int(counterarray[img[0]])))
        counterarray[img[0]] += 1
        if i%50==0:
            print(i)
        i = i+1
        if i==N_max:
            break
