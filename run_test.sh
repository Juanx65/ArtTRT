#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh
echo "# resnet152 bs 1"
echo " "
./main.sh 1 resnet152 build
#rm weights/*.engine
#rm weights/*.onnx
echo " "