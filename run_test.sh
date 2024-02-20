#!/bin/bash

# remeber to do a chmod +x run_test.sh befor runing with ./run_test.sh
echo "# resnet152 bs 256"
echo " "
./main.sh 256 resnet152
rm weights/*.engine
rm weights/*.onnx
echo " "