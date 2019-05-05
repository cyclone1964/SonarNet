# This script evaluates a sonar model. A lot of it is cut and pasted
# from sonareNet.py and frankly should probably be better managed but
# it is what it is at this point.

# The first thing is to import all the things we need
#
# os, re, and math for file manipulation
# pytorch of course
# numpy for data input manipulations
# matplotlib for plotting intermediates
# shorthands for nn and model_zoo
# The data set and data loader support
import os
import re
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Now, we write our own data set class so that we can leverage all
# loader capabilities of pytorch. The structure here is that the
# generated data is stored in a directory in a series of files with a
# specific naming convention.
#
# Directory.txt - list of all the IDs
# ImageMap-<ID>.txt - file containing the features, 25x128x256 greyscale
# LabelMap-<ID>.txt - file containing flags of 1/0 about target presence
# Detections-<ID>.txt - file containg flags for each of the frames/bins
# FeatureMap-<ID>.txt - 8 sets of X/Y/Z/Doppler/Validity for targets
#
# For this processing, we are only interested in the ImageMaps and Detections
class SonarDataset(Dataset):

    """ Sonar Dataset: for reading my generated Sonar data """
    def __init__(self, root_dir,partition = 'train'):
        """
        Args:
             root_dir (string): path to the data directory
        
        This routine reads in the vector of IDs from the Directory
        and sets up index lists for the various partitions
        """
        self.root_dir = root_dir

        # This loads the directory, which is a list of all viable IDs
        self.directory = np.loadtxt(root_dir + '/Directory.txt').astype(int)

        # Now splut that into training, validation, and test. Training
        # is the first 1/2, validation the next 1/4, and test the rest
        numTrain = math.floor(len(self.directory)/2)
        numVal = math.floor(len(self.directory)/4)
        self.partitions = {'train': range(0,numTrain),
                           'validate': range(numTrain,numTrain+numVal),
                           'test': range(numTrain+numVal,len(self.directory))}

        # Now for this instantiation, we choose one of the partitions
        self.indices = self.partitions[partition]; 

    """
    Return the length of the current partition
    """
    def __len__(self):
        return len(self.indices)

    """ 
    Get an instance of the current partition. The input is taken as an
    index into the currently selected index and the associated
    files are read in.
    """
    def __getitem__(self,idx):

        # Get the directory index for this partition location
        index = self.indices[idx]

        # Now load the image, reshape as necessary, and convert to a
        # torch tensor of floats.
        X = np.fromfile(self.root_dir + '/ImageMap-' +
                        str(self.directory[index]) + '.dat',
                       dtype='uint8').astype(float)
        X.shape = (25, 128, 256)
        X = torch.from_numpy(X).float()

        # And similarly the detections ...
        y = np.fromfile(self.root_dir + '/Detections-' +
                        str(self.directory[index]) + '.dat',
                       dtype='uint8').astype(float)
        y = torch.from_numpy(y).float()
        
        return X, y

# Now define the SonarNet, a pytorch nn module that is pretty much a
# clone of VGG 16 but with some dimensions changed
class SonarNet(nn.Module):

    def __init__(self):

        # Initialize the parent object
        super(SonarNet, self).__init__()

        # The "Front end", which consists of two pairs of VGG style
        # small convolutions, each constrained to only it's individual
        # plane (which is to say sonar beam), followed by two more
        # that span the beams and provide for inter-beam learning
        # opportunities.
        
        self.frontEnd = nn.Sequential(
            ## This is almost cut and paste from VGG16 except that the
            ## number of input planes is radically different.
            nn.Conv2d(25,64, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,64, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0,
                         dilation=1,ceil_mode=False),
            
            nn.Conv2d(64,128, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0,
                         dilation=1,ceil_mode=False),

            nn.Conv2d(128,256, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0,
                         dilation=1,ceil_mode=False),

            nn.Conv2d(256,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0,
                         dilation=1,ceil_mode=False),

            nn.Conv2d(512,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,512, kernel_size=(3,3),stride=(1,1),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0,
                         dilation=1,ceil_mode=False),
            
        )
        
        # And now the classifier layers, two fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 4 * 8, 128+256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128+256, 128+256),
            nn.Sigmoid()
        )

    # Now the forward propgation. This runs the front end, then feeds
    # the output of that to the range and frequency nets, concatenates
    # them, and makes the output.
    def forward(self, x):
        
        # Run the front end
        front = self.frontEnd(x)

        # Flatten it
        flat = front.view(front.size(0),512 * 4 * 8)
        
        # Run the classifier on them both
        x = self.classifier(flat)

        return x

# Set up the data set
dataDir = '../GeneratedData'
trainingSet = SonarDataset(dataDir,partition = 'train')
validationSet = SonarDataset(dataDir,partition = 'validate')
validationLoader = DataLoader(validationSet,batch_size=10)

# Create the sonarnet, insuring that it is implemented in floats not
# doubles since that is how it was saved in the file
model = SonarNet().float()

fileName = 'CheckPoint-116.pth'
checkPoint = torch.load(fileName,map_location = "cpu")

# Now extract the things from the dicionary
epoch = checkPoint['epoch']
model.load_state_dict(checkPoint['state_dict'])
losses = checkPoint['losses']
valPerformance = checkPoint['valPerformance']
trainPerformance = checkPoint['trainPerformance']

# Now let us do the validation
numTotal = 0
numCorrect = 0
for X_val, y_val in validationLoader:

    y_pred = model.forward(X_val)

    predBinValues,predBinIndices = torch.max(y_pred[:,1:256],dim=1)
    predFrameValues,predFrameIndices = torch.max(y_pred[:,256:],dim=1)
    print("predBinValues: ",predBinValues)
    print("predBinIndices: ",predBinIndices)
    print("predFrameValues: ",predFrameValues)
    print("predFrameIndices: ",predFrameIndices)
    
    valBinValues,valBinIndices = torch.max(y_val[:,1:256],dim=1)
    valFrameValues,valFrameIndices = torch.max(y_val[:,256:],dim=1)
    print("valBinValues: ",valBinValues)
    print("valBinIndices: ",valBinIndices)
    print("valFrameValues: ",valFrameValues)
    print("valFrameIndices: ",valFrameIndices)

    #  The mode is correct iff:
    #
    # There is no target and it found no target or
    # there is a target and the model found it in the right place
    states = (torch.gt(predBinValues,0.5) & 
              torch.gt(predFrameValues,0.5) & 
              torch.gt(valBinValues,0.5) & 
              torch.gt(valFrameValues,0.5) & 
              torch.eq(predBinIndices,valBinIndices) &
              torch.eq(predFrameIndices,valFrameIndices)).cpu().numpy()
    print("Det States: ",states);
    print("Correct: ",np.sum(np.where(states,1,0)));
    numDets = np.sum(np.where(states,1,0))
    numCorrect += numDets

    states = (torch.le(predBinValues,0.5) &
              torch.le(predFrameValues,0.5) & 
              torch.le(valBinValues,0.5) & 
              torch.le(valFrameValues,0.5)).cpu().numpy()
    print("No States: ",states);
    print("Correct: ",np.sum(np.where(states,1,0)));
    print("Total: ",states.shape[0])
    numNo = np.sum(np.where(states,1,0))
    numCorrect += numNo
    numTotal += states.shape[0]
    print(" Stats: ",numDets,",",numNo,",",numCorrect)

print("Performance: ",float(numCorrect)/float(numTotal))
