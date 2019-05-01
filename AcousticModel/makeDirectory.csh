#!/bin/tcsh
cd ../GeneratedData
ls | grep Image | sed -e 's/ImageMap-//g' | sed -e 's/.dat//g' >! Directory.txt
