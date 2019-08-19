#!/bin/zsh
cd ../GeneratedData/train
ls | grep Image | sed -e 's/ImageMap-//g' | sed -e 's/.dat//g' >! Directory.txt
