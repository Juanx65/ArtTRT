#!/bin/bash

# remeber to do a chmod +x build_experiment.sh befor runing with ./build_experiment.sh
BATCH_SIZE=$1
NETWORK=$2

C=3
W=224
H=224
 
#TRT FP32
echo "1er"
#TRT FP32
python onnx_transform.py --weights="weights/best_fp32.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W  #.pt para yolo
python build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W 
python utils/experiments/get_parameters.py --weights weights/best_fp32.pth --engine weights/best_fp32.engine -trt --verbose  --network="$NETWORK" 
python utils/experiments/get_parameters.py --weights weights/best_fp32.pth --engine weights/best_fp32.engine --verbose  --network="$NETWORK"


rm weights/*.onnx > /dev/null 2>&1
rm weights/*.engine > /dev/null 2>&1
#TRT FP32
echo "2do"
python onnx_transform.py --weights="weights/best_fp32.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python utils/experiments/get_parameters.py --engine weights/best_fp32.engine -trt

rm weights/*.onnx > /dev/null 2>&1
rm weights/*.engine > /dev/null 2>&1
#TRT FP32
echo "3ero"
python onnx_transform.py --weights="weights/best_fp32.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python utils/experiments/get_parameters.py --engine weights/best_fp32.engine -trt 

rm weights/*.onnx > /dev/null 2>&1
rm weights/*.engine > /dev/null 2>&1
#TRT FP32
echo "4to"
python onnx_transform.py --weights="weights/best_fp32.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
python utils/experiments/get_parameters.py --engine weights/best_fp32.engine -trt

rm weights/*.onnx > /dev/null 2>&1
rm weights/*.engine > /dev/null 2>&1