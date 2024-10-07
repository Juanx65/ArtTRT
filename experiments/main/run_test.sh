#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh

# main.sh test
echo "# Orin AGX PM2 LVL5"

echo " "
echo "## resnet50 bs 1"
echo " "
./experiments/main/main.sh 1 resnet50 build dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 32"
echo " "
./experiments/main/main.sh 32 resnet50 build dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 64"
echo " "
./experiments/main/main.sh 64 resnet50 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 128"
echo " "
./experiments/main/main.sh 128 resnet50 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet50 bs 256"
echo " "
./experiments/main/main.sh 256 resnet50 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet152 bs 1"
echo " "
./experiments/main/main.sh 1 resnet152 build dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet152 bs 32"
echo " "
./experiments/main/main.sh 32 resnet152 build dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet152 bs 64"
echo " "
./experiments/main/main.sh 64 resnet152 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet152 bs 128"
echo " "
./experiments/main/main.sh 128 resnet152 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats

echo " "
echo "## resnet152 bs 256"
echo " "
./experiments/main/main.sh 256 resnet152 nonbuild dynamic 5 datasets/subdataset_val/val PM2 tegrastats