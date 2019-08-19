#!/bin/zsh
cd ../GeneratedData/test
ls | grep Image | sed -e 's/ImageMap-//g' | sed -e 's/.dat//g' >! Directory.txt
