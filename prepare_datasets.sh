#!/bin/bash

echo "Creating datasets folder"
mkdir datasets
cd datasets

#################################

mkdir HARTH

if ! [ -f ./harth.zip ]; then
    echo "Downloading HARTH"
    wget http://www.archive.ics.uci.edu/static/public/779/harth.zip
fi
if ! [ "$(ls -A ./HARTH)" ]; then
    unzip harth.zip -d ./HARTH
else
    echo "HARTH directory not empty"
fi

#################################

mkdir UCI-HAPT

if ! [ -f ./smartphone+based+recognition+of+human+activities+and+postural+transitions.zip ]; then
    echo "Downloading UCI-HAPT"
    wget https://archive.ics.uci.edu/static/public/341/smartphone+based+recognition+of+human+activities+and+postural+transitions.zip
fi
if ! [ "$(ls -A ./UCI-HAPT)" ]; then
    unzip smartphone+based+recognition+of+human+activities+and+postural+transitions.zip -d ./UCI-HAPT
else
    echo "UCI-HAPT directory not empty"
fi

#################################

mkdir WISDM

if ! [ -f ./WISDM_ar_latest.tar.gz ]; then
    echo "Downloading WISDM"
    wget https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz
fi
if ! [ "$(ls -A ./WISDM)" ]; then
    tar -xvf WISDM_ar_latest.tar.gz -C ./WISDM
else
    echo "WISDM directory not empty"
fi
