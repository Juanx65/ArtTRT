#!/bin/bash

# remeber to do a chmod +x gen_tables.sh befor runing with ./gen_tables.sh

echo "mobilenet bs 1"
./main_all.sh 1 mobilenet
echo "=================="
echo "mobilenet bs 32"
./main_all.sh 32 mobilenet
echo "=================="
echo "mobilenet bs 64"
./main_all.sh 64 mobilenet
echo "=================="
echo "mobilenet bs 128"
./main_all.sh 128 mobilenet
echo "=================="
echo "mobilenet bs 256"
./main_all.sh 256 mobilenet
echo "=================="

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

echo "resnet34 bs 1"
./main_all.sh 1 resnet34
echo "=================="
echo "resnet34 bs 32"
./main_all.sh 32 resnet34
echo "=================="
echo "resnet34 bs 64"
./main_all.sh 64 resnet34
echo "=================="
echo "resnet34 bs 128"
./main_all.sh 128 resnet34
echo "=================="
echo "resnet34 bs 256"
./main_all.sh 256 resnet34
echo "=================="

echo "resnet50 bs 1"
./main_all.sh 1 resnet50
echo "=================="
echo "resnet50 bs 32"
./main_all.sh 32 resnet50
echo "=================="
echo "resnet50 bs 64"
./main_all.sh 64 resnet50
echo "=================="
echo "resnet50 bs 128"
./main_all.sh 128 resnet50
echo "=================="
echo "resnet50 bs 256"
./main_all.sh 256 resnet50
echo "=================="

echo "resnet101 bs 1"
./main_all.sh 1 resnet101
echo "=================="
echo "resnet101 bs 32"
./main_all.sh 32 resnet101
echo "=================="
echo "resnet101 bs 64"
./main_all.sh 64 resnet101
echo "=================="
echo "resnet101 bs 128"
./main_all.sh 128 resnet101
echo "=================="
echo "resnet101 bs 256"
./main_all.sh 256 resnet101
echo "=================="

echo "resnet152 bs 1"
./main_all.sh 1 resnet152
echo "=================="
echo "resnet152 bs 32"
./main_all.sh 32 resnet152
echo "=================="
echo "resnet152 bs 64"
./main_all.sh 64 resnet152
echo "=================="
echo "resnet152 bs 128"
./main_all.sh 128 resnet152
echo "=================="
echo "resnet152 bs 256"
./main_all.sh 256 resnet152
echo "=================="