#!/bin/tcsh
set fileName=CheckPoint-$1.pth
scp matthew_daily@seawulf.uri.edu:/home/matthew_daily/SonarNet/NeuralNet/$fileName $fileName
