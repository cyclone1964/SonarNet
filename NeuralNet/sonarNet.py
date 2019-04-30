# The first thing is to import all the things we need
#
# pytorch of course
# numpy for data input manipulations
# matplotlib for plotting intermediates
# shorthands for nn and model_zoo
# The data set and data loader support
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
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
# FeatureMap-<ID>.txt - 8 sets of X/Y/Z/Doppler/Validity for targets
#
# The ImageMaps we used unchanged
#
# The LabelMaps provide the labels. but are reduced from 128-256 to
# two catenated vectors 128 adn 256 by or-ing accross the two
# dimensions. This was done to reduce the size of the net.
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
        self.partitions = {'train': range(0,15000),
                           'validate': range(15000,20000),
                           'test': range(20000,25000)}
        self.directory = np.loadtxt(root_dir + '/Directory.txt').astype(int)
        self.indices = self.partitions[partition]; 

    """
    Return the length of the current partition
    """
    def __len__(self):
        return len(self.indices)

    """ 
    Get an instance of the current partition. The input is taken as an
    index into the currently selected index list and the associated
    files are read in. It is at this point that the reduction in the
    dimensionality of the labels is done.
    """
    def __getitem__(self,idx):

        # Get the directory index for this partition location
        index = self.indices[idx]

        # Now load the image, reshape as necessary, and convert to a
        # torch tensor.
        X = np.fromfile(self.root_dir + '/Image-' +
                        str(self.directory[index]) + '.csv',
                       dtype='uint8').astype(float)
        X.shape = (25, 128, 256)
        X = torch.from_numpy(X).float()

        # And similarly the labels ...
        y = np.fromfile(self.root_dir + '/LabelMap-' +
                        str(self.directory[index]) + '.csv',
                       dtype='uint8').astype(float)
        y.shape = (128,256)
        
        # ... and do the reduction by projecting onto range and frequency
        # axes and concatenating
        range = y.max(1)
        freq = y.max(0)
        range.shape = (128,)
        freq.shape = (256,)
        y = np.concatenate((range,freq))
        y = torch.from_numpy(y).float()
        
        return X, y

# Now define the SonarNet, a pytorch nn module.
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
            # This I chose because I feel like the echoes are contained in 
            # a fairly tight space along the frequency axis, but they 
            # can be right at the edge so I do a full padding here. 
            # The output will be 25x128x256
            nn.Conv2d(25,25, kernel_size=(3,3),padding = (1,1),groups=25),
            nn.ReLU(inplace=True),
            nn.Conv2d(25,25, kernel_size=(3,3),padding = (1,1),groups=25),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
            
            nn.Conv2d(25,25, kernel_size=(3,3),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(25,25, kernel_size=(3,3),padding = (1,1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1,padding=1),
        )
        
        # This is the "RangeNet", which expands the input (25x128x256) out to
        # (128x128x256) then does a big maxPool over the entire map, generating
        # a 1x128 vector that is matched to the range part of the labels.
        self.rangeNet = nn.Sequential(
            nn.Conv2d(25,128,kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(128,256)),
        )
        
        # Similarly the frequencyNet
        self.frequencyNet = nn.Sequential(
            nn.Conv2d(25,256,kernel_size=11),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(256,256),)
        )
        
        
        # And now the classifier layers, two fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128+256, 128+256),
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
        
        # now run the freqency and range nets
        range = self.rangeNet(front)
        frequency = self.frequencyNet(front)

        # Now to catenate them together
        range = range.view(range.size(0),128)
        frequency = frequency.view(frequency.size(0),256)
        both = torch.cat((range,frequency),1)

        # Run the classifier on them both
        x = self.classifier(both)

        return x

# Now, we test the data loader by loading one item and plotting it
dataDir = '../GeneratedData'
trainingSet = SonarDataset(dataDir,partition = 'train')

if (!torch.cude.is_available()):
    X, y = trainingSet.__getitem__(5)
    print("X: ",X.shape,"Y: ",y.shape)
    plt.figure
    plt.imshow(X[13,:,:])
    plt.figure
    plt.plot(y[:128].numpy())
    plt.figure
    plt.plot(y[128:].numpy())


# Now, let's try to train a network!! First, set up the data loader
# from the training set above with a batch size of 5 for now.
loader = DataLoader(trainingSet,batch_size = 5)

# Create the sonarnet, insuring that it is implemented in floats not
# doubles since it runs faster.
model = SonarNet().float()
model.to(device)

# Use simple SGD to start
optim = torch.optim.SGD(model.parameters(),lr=1e-2)

# Now, we do some training. First, we have the option of loading from
# a checkpoint.
fileName = "CheckPoint-860.pth"
if (os.path.isfile(fileName)):
    print("Loading Checkpoint: ",fileName)
    checkpoint = torch.load(fileName)
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optim.load_state_dict(checkpoint['optimizer'])
    losses = checkpoint['losses']
else:
    print("Checkpoint File not Found: ",fileName);
    epoch = 0
    losses = []

# Now do the epochs
while (epoch < 5):
    print("Do Epoch: ",epoch)

    for X_batch, y_batch in loader:
        X.batch.to(device)
        y.batch.to(device)
        y_pred = model.forward(X_batch)
        loss = torch.nn.functional.mse_loss(y_pred,y_batch)

        loss.backward()
        optim.step()
        optim.zero_grad()

        losses.append(loss.item())
        print("Index: ", len(losses), " Loss: ",loss.item())
            
        # Now do the plotting
        
        if (len(losses)%20 == 0):
            fileName = 'CheckPoint-' + str(len(losses)) +'.pth'
            state = {'losses': losses,
                     'epoch': epoch,
                     'state_dict':model.state_dict(),
                     'optimizer': optim.state_dict()}
            torch.save(state,fileName)
            print('Saved Checkpoint: ',fileName)
