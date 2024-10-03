#!/bin/bash

# remeber to do a chmod +x static_batch_test.sh befor runing with ./static_batch_test.sh

# main_static.sh test
echo "# Orin AGX PM2"

echo " "
echo "## resnet50 bs 1"
echo " "
./experiments/main/main_static.sh 1 resnet50 build datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "# resnet50 bs 32"
echo " "
./experiments/main/main_static.sh 32 resnet50 build datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 64"
echo " "
./experiments/main/main_static.sh 64 resnet50 build datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 128"
echo " "
./experiments/main/main_static.sh 128 resnet50 build datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 256"
echo " "
./experiments/main/main_static.sh 256 resnet50 build datasets/subdataset_val/val PM2 tegrastats