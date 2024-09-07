#!/bin/bash

PM=$1

set -e
echo " "
echo "# mobilenet bs 1"
echo " "
./experiments/main/main.sh 1 mobilenet build datasets/subdataset_val/val --profile $PM

echo " "
echo "# mobilenet bs 32"
echo " "
./experiments/main/main.sh 32 mobilenet build datasets/subdataset_val/val --profile $PM

echo " "
echo "# mobilenet bs 64"
echo " "
./experiments/main/main.sh 64 mobilenet nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# mobilenet bs 128"
echo " "
./experiments/main/main.sh 128 mobilenet nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# mobilenet bs 256"
echo " "
./experiments/main/main.sh 256 mobilenet nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet50 bs 1"
echo " "
./experiments/main/main.sh 1 resnet50 build datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet50 bs 32"
echo " "
./experiments/main/main.sh 32 resnet50 build datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet50 bs 64"
echo " "
./experiments/main/main.sh 64 resnet50 nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet50 bs 128"
echo " "
./experiments/main/main.sh 128 resnet50 nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet50 bs 256"
echo " "
./experiments/main/main.sh 256 resnet50 nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet152 bs 1"
echo " "
./experiments/main/main.sh 1 resnet152 build datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet152 bs 32"
echo " "
./experiments/main/main.sh 32 resnet152 build datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet152 bs 64"
echo " "
./experiments/main/main.sh 64 resnet152 nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet152 bs 128"
echo " "
./experiments/main/main.sh 128 resnet152 nonbuild datasets/subdataset_val/val --profile $PM

echo " "
echo "# resnet152 bs 256"
echo " "
./experiments/main/main.sh 256 resnet152 nonbuild datasets/subdataset_val/val --profile $PM
set +e