#!/bin/tcsh
cd /Users/Matt/Documents/URI/Classes
cd CSC\ 592\ Deep\ Learning/Project/SonarNet/GeneratedData
cat Directory.txt | head -1000 | \
awk '{print "ImageMap-" $0 ".dat"; print "Detections-" $0 ".dat";}' >! File.txt
tar cvf DataSets.tar -T File.txt
scp DataSets.tar \
matthew_daily@seawulf.uri.edu:/home/matthew_daily/SonarNet/GeneratedData
rm -f DataSets.tar File.txt
