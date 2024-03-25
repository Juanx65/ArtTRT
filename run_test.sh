#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh
echo "# resnet18 bs 1"
echo " "
./main.sh 1 resnet18 nonbuild
#rm weights/*.engine
#rm weights/*.onnx
echo " "