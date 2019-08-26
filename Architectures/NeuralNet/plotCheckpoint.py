# A simple script that reads in a checkpoint as I have defined it and
# plots the losses in that checkpoint.
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

fileName = 'CheckPoint-' + sys.argv[1] + '.pth'
checkpoint = torch.load(fileName,map_location="cpu")

losses = checkpoint['losses']
valPerformance = checkpoint['valPerformance']
trainPerformance = checkpoint['trainPerformance']
epoch = checkpoint['epoch']

plt.figure()
plt.plot(losses)
plt.title(fileName + ": " + str(epoch))

plt.figure()
plt.plot(valPerformance,'b')
plt.plot(trainPerformance,'k')
plt.title('Performance: ' + fileName)
plt.show()

