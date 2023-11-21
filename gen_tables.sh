#!/bin/bash

# remeber to do a chmod +x gen_tables.sh befor runing with ./gen_tables.sh

echo "# mobilenet bs 1"
echo " " 
./main.sh 1 mobilenet
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# mobilenet bs 32"
echo " " 
./main.sh 32 mobilenet
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# mobilenet bs 64"
echo " " 
./main.sh 64 mobilenet
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# mobilenet bs 128"
echo " " 
./main.sh 128 mobilenet
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# mobilenet bs 256"
echo " " 
./main.sh 256 mobilenet
rm weights/*.engine
rm weights/*.onnx
echo " "

echo "# resnet18 bs 1"
echo " "
./main.sh 1 resnet18
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet18 bs 32"
echo " "
./main.sh 32 resnet18
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet18 bs 64"
echo " "
./main.sh 64 resnet18
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet18 bs 128"
echo " "
./main.sh 128 resnet18
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet18 bs 256"
echo " "
./main.sh 256 resnet18
rm weights/*.engine
rm weights/*.onnx
echo " "

echo "# resnet34 bs 1"
echo " "
./main.sh 1 resnet34
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet34 bs 32"
echo " "
./main.sh 32 resnet34
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet34 bs 64"
echo " "
./main.sh 64 resnet34
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet34 bs 128"
echo " "
./main.sh 128 resnet34
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet34 bs 256"
echo " "
./main.sh 256 resnet34
rm weights/*.engine
rm weights/*.onnx
echo " "

echo "# resnet50 bs 1"
echo " "
./main.sh 1 resnet50
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet50 bs 32"
echo " "
./main.sh 32 resnet50
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet50 bs 64"
echo " "
./main.sh 64 resnet50
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet50 bs 128"
echo " "
./main.sh 128 resnet50
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet50 bs 256"
echo " "
./main.sh 256 resnet50
rm weights/*.engine
rm weights/*.onnx
echo " "

echo "# resnet101 bs 1"
echo " "
./main.sh 1 resnet101
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet101 bs 32"
echo " "
./main.sh 32 resnet101
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet101 bs 64"
echo " "
./main.sh 64 resnet101
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet101 bs 128"
echo " "
./main.sh 128 resnet101
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet101 bs 256"
echo " "
./main.sh 256 resnet101
rm weights/*.engine
rm weights/*.onnx
echo " "

echo "# resnet152 bs 1"
echo " "
./main.sh 1 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet152 bs 32"
echo " "
./main.sh 32 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet152 bs 64"
echo " "
./main.sh 64 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet152 bs 128"
echo " "
./main.sh 128 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "
echo "# resnet152 bs 256"
echo " "
./main.sh 256 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "