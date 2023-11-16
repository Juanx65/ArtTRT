#!/bin/bash

# remeber to do a chmod +x gen_tables.sh befor runing with ./gen_tables.sh

echo "resnet18 bs 1"
./main_all.sh 1 resnet18
echo "=================="
echo "resnet18 bs 32"
./main_all.sh 32 resnet18
echo "=================="
echo "resnet18 bs 64"
./main_all.sh 64 resnet18
echo "=================="
echo "resnet18 bs 128"
./main_all.sh 128 resnet18
echo "=================="
echo "resnet18 bs 256"
./main_all.sh 256 resnet18
echo "=================="