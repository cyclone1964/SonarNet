# A simple script that reads in a checkpoint as I have defined it and
# plots the losses in that checkpoint.
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

fileName = 'CheckPoint-6080.pth'
checkpoint = torch.load(fileName,map_location="cpu")

losses = checkpoint['losses']

plt.figure()
plt.plot(losses)
plt.show()

