#!/bin/tcsh
if ($# < 1) then
    echo Usage $0 Size
    exit(1)
endif

# Move to the data directory
cd ../GeneratedData

# Form a list of the file to be tarred up by catting the directory
cat Directory.txt | head -$1 | \
awk '{print "ImageMap-" $0 ".dat"; print "Detections-" $0 ".dat";}' >! File.txt

# Tar them up
tar cvf DataSets.tar -T File.txt

# Now copy them to the remove server
scp DataSets.tar \
matthew_daily@seawulf.uri.edu:/home/matthew_daily/SonarNet/GeneratedData
rm -f DataSets.tar File.txt
