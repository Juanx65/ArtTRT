#!/bin/bash

# remeber to do a chmod +x gen_tables.sh befor runing with ./gen_tables.sh

# generate all the tables for the plataform tested.

echo "# mobilenet bs 1"
echo " " 
./main.sh 1 mobilenet build 

echo " "
echo "# mobilenet bs 32"
echo " " 
./main.sh 32 mobilenet build

echo " "
echo "# mobilenet bs 64"
echo " " 
./main.sh 64 mobilenet nonbuild

echo " "
echo "# mobilenet bs 128"
echo " " 
./main.sh 128 mobilenet nonbuild

echo " "
echo "# mobilenet bs 256"
echo " " 
./main.sh 256 mobilenet nonbuild

echo " "
echo "# resnet18 bs 1"
echo " "
./main.sh 1 resnet18 build

echo " "
echo "# resnet18 bs 32"
echo " "
./main.sh 32 resnet18 build

echo " "
echo "# resnet18 bs 64"
echo " "
./main.sh 64 resnet18 nonbuild

echo " "
echo "# resnet18 bs 128"
echo " "
./main.sh 128 resnet18 nonbuild

echo " "
echo "# resnet18 bs 256"
echo " "
./main.sh 256 resnet18 nonbuild

echo " "
echo "# resnet34 bs 1"
echo " "
./main.sh 1 resnet34 build

echo " "
echo "# resnet34 bs 32"
echo " "
./main.sh 32 resnet34 build

echo " "
echo "# resnet34 bs 64"
echo " "
./main.sh 64 resnet34 nonbuild

echo " "
echo "# resnet34 bs 128"
echo " "
./main.sh 128 resnet34 nonbuild

echo " "
echo "# resnet34 bs 256"
echo " "
./main.sh 256 resnet34 nonbuild

echo " "
echo "# resnet50 bs 1"
echo " "
./main.sh 1 resnet50 build

echo " "
echo "# resnet50 bs 32"
echo " "
./main.sh 32 resnet50 build

echo " "
echo "# resnet50 bs 64"
echo " "
./main.sh 64 resnet50 nonbuild

echo " "
echo "# resnet50 bs 128"
echo " "
./main.sh 128 resnet50 nonbuild

echo " "
echo "# resnet50 bs 256"
echo " "
./main.sh 256 resnet50 nonbuild

echo " "
echo "# resnet101 bs 1"
echo " "
./main.sh 1 resnet101 build

echo " "
echo "# resnet101 bs 32"
echo " "
./main.sh 32 resnet101 build

echo " "
echo "# resnet101 bs 64"
echo " "
./main.sh 64 resnet101 nonbuild

echo " "
echo "# resnet101 bs 128"
echo " "
./main.sh 128 resnet101 nonbuild

echo " "
echo "# resnet101 bs 256"
echo " "
./main.sh 256 resnet101 nonbuild

echo " "
echo "# resnet152 bs 1"
echo " "
./main.sh 1 resnet152 build

echo " "
echo "# resnet152 bs 32"
echo " "
./main.sh 32 resnet152 build

echo " "
echo "# resnet152 bs 64"
echo " "
./main.sh 64 resnet152 nonbuild

echo " "
echo "# resnet152 bs 128"
echo " "
./main.sh 128 resnet152 nonbuild

echo " "
echo "# resnet152 bs 256"
echo " "
./main.sh 256 resnet152 nonbuild

echo " "