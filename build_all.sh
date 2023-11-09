#!/bin/bash

# remeber to do a chmod +x build_all.sh befor runing with ./build_all.sh
BATCH_SIZE=1
NETWORK="resnet18"

C=3
W=224
H=224

#TRT FP32
python3 onnx_transform.py --weights="weights/best_fp32.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W
python3 build_trt.py --weights="weights/best_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W

#TRT FP16
python3 onnx_transform.py --weights="weights/best_fp16.pth" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W
python3 build_trt.py --weights="weights/best_fp16.onnx"  --fp16 --input_shape $BATCH_SIZE $C $H $W