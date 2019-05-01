#!/bin/tcsh
cd /Users/Matt/Documents/URI/Classes
cd CSC\ 592\ Deep\ Learning/Project/SonarNet/GeneratedData
ls | grep Image | sed -e 's/ImageMap-//g' | sed -e 's/.dat//g' >! Directory.txt
