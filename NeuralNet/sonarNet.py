# This version implements a conv/deconv layer to do the detection

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

# First, let's see if we have a cuda device
if torch.cuda.is_available():
    print("Run on GPU!!");
    device = torch.device("cuda:0")
else:
    print("Run on CPU :(");
    device = torch.device("cpu")

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
        X.shape = (25, 64, 64)
        X = torch.from_numpy(X).float()

        # And similarly the detections ...
        y = np.fromfile(self.root_dir + '/Detections-' +
                        str(self.directory[index]) + '.dat',
                       dtype='uint8').astype(float)
        y = np.minimum(np.maximum(y,0.25),0.75)
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
        # opportunities. The inputs and outpus are 25x64x64
        self.frontEnd = nn.Sequential(
            ## This is almost cut and paste from VGG16 except that the
            ## number of input planes is radically different.
            nn.Conv2d(25,25, groups = 25,
                      kernel_size=(7,7),stride=(1,1),padding = (3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=5, stride=1,padding=2,
                         dilation=1,ceil_mode=False),

            nn.Conv2d(25,25,
                      kernel_size=(11,11),stride=(1,1),padding = (5,5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1,
                         dilation=1,ceil_mode=False)
        )

        # now plane to get the frame flags
        self.framePlane = nn.Sequential(
            nn.Conv2d(25,64,
                      kernel_size=(11,11),stride=(1,1),padding = (5,5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (64,64),stride = (1,1),padding=0)
        )
                      
        self.binPlane = nn.Sequential(
            nn.Conv2d(25,64,
                      kernel_size=(11,11),stride=(1,1),padding = (5,5)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (64,64),stride = (1,1),padding=0)
        )

        # And now the classifier layers, two fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128,128)
        )

    # Now the forward propgation. This runs the front end, then feeds
    # the output of that to the range and frequency nets, concatenates
    # them, and makes the output.
    def forward(self, x):
        
        # First, lets do the three sets of convolutions
        front = self.frontEnd(x)

        frames = self.framePlane(front)
        bins = self.binPlane(front)

        # Flatten and classify
        frames=frames.view(frames.size(0),64)
        bins=bins.view(bins.size(0),64)

        flat=torch.cat((bins,frames),1)
        
        # Run the classifier on them both
        y = self.classifier(flat)
        return y

# Set up the data set
dataDir = '../GeneratedData'
trainingSet = SonarDataset(dataDir,partition = 'train')
validationSet = SonarDataset(dataDir,partition = 'validate')

# Now, let's try to train a network!! First, set up the data loader
# from the training set above with a batch size of 20 for now.
trainingLoader = DataLoader(trainingSet,batch_size = 20)
validationLoader = DataLoader(validationSet,batch_size = 20)

# Create the sonarnet, insuring that it is implemented in floats not
# doubles since it runs faster.
model = SonarNet().float()
model = model.to(device)

# Use simple SGD to start
optim = torch.optim.SGD(model.parameters(),lr=1e-2,momentum=0.2)

# Now, we do some training. First, we have the option of loading from
# a checkpoint if we find one: start from the highest numbered one
lastCheckpoint = 0
for r,d,f in os.walk('.'):
    for file in f:
        if 'CheckPoint' in file:
            parts = re.split("[-\.]",file)
            lastCheckpoint = max(lastCheckpoint,int(parts[1]))
            print("Found Checkpoint",lastCheckpoint)

# If we found a checkpoint, load it, otherwise, start from scratch
if (lastCheckpoint > 0):
    fileName = "CheckPoint-" + str(lastCheckpoint) + ".pth"
    print("Loading Checkpoint: ",fileName)
    if (torch.cuda.is_available()):
        checkpoint = torch.load(fileName)
    else:
        checkpoint = torch.load(fileName,map_location = "cpu")

    # Now extract the things from the dicionary
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    losses = checkpoint['losses']
    if ('valPerformance' in checkpoint):
        valPerformance = checkpoint['valPerformance']
    else:
        valPerformance = []

    if ('trainPerformance' in checkpoint):
        trainPerformance = checkpoint['trainPerformance']
    else:
        trainPerformance = []
        
else:
    print("Checkpoint Not Found");
    epoch = 0
    losses = []
    valPerformance = []
    trainPerformance = []
    
# Now do up to 2000 epochs
while (epoch < 2000):

    # For all the batches
    numTotal=0
    numCorrect = 0
    for X_batch, y_batch in trainingLoader:

        # Send them to the cuda device if we are using one
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        # DO the preduction and add the loss
        y_pred = model.forward(X_batch)
        loss = torch.nn.functional.mse_loss(y_pred,y_batch)
        
        # Back propagate
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        # Append the loss for this to the list
        losses.append(loss.item())

        predBinValues,predBinIndices = torch.max(y_pred,dim=1)
        batchBinValues,batchBinIndices = torch.max(y_batch,dim=1)

        #  The mode is correct iff the bin values match
        states = (torch.eq(predBinIndices,batchBinIndices)).cpu().numpy()
        numCorrect += np.sum(np.where(states,1,0))
        numTotal += states.shape[0]

    # Now, the first time through , we have to have an extra 
    trainPerformance.append(float(numCorrect)/float(numTotal))
    
    # Now let us do the validation
    numTotal = 0
    numCorrect = 0
    for X_val, y_val in validationLoader:

        X_val = X_val.to(device)
        y_val = y_val.to(device)
        y_pred = model.forward(X_val)
        
        predBinValues,predBinIndices = torch.max(y_pred,dim=1)
        valBinValues,valBinIndices = torch.max(y_val,dim=1)

        # The mode is correct iff the bin value right
        states = (torch.eq(predBinIndices,valBinIndices)).cpu().numpy()
        numCorrect += np.sum(np.where(states,1,0))
        numTotal += states.shape[0]

    # Add this to the performance numbers
    valPerformance.append(float(numCorrect)/float(numTotal))
    
    # Now, every 1000 or so, we save a checkpoint so that we can
    # restart from there.
    fileName = 'CheckPoint-' + str(epoch) +'.pth'
    state = {'epoch': epoch,
             'losses': losses,
             'trainPerformance':trainPerformance, 
             'valPerformance':valPerformance, 
             'state_dict' : model.state_dict(),
             'optimizer': optim.state_dict()}
    torch.save(state,fileName)
    print('Saved Checkpoint ',fileName,
          ' train: ',trainPerformance[-1],
          ' val:',valPerformance[-1])

    # Next epoch please
    epoch = epoch + 1

