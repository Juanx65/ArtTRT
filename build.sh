#!/bin/bash

# remeber to do a chmod +x build.sh befor runing with ./build.sh
BATCH_SIZE=$1 #128
NETWORK=$2 #"resnet18"

C=3
W=224
H=224

echo $INPUT_SHAPE

#ONNX python onnx_transform.py --weights weights/best.pth --pretrained --network mobilenet --input_shape 1 3 224 224
python onnx_transform.py --weights="weights/best.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W

#TRT FP32
python build_trt.py --weights="weights/best.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp32.engine

#TRT FP16
python build_trt.py --weights="weights/best.onnx"  --fp16 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp16.engine

#TRT INT8
rm outputs/cache/*.cache
python build_trt.py --weights="weights/best.onnx"  --int8 --input_shape $BATCH_SIZE $C $H $W --engine_name best_int8.engine