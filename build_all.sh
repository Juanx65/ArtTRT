#!/bin/bash

# remeber to do a chmod +x build_all.sh befor runing with ./build_all.sh
BATCH_SIZE=1
NETWORK="yolo"

C=3
W=224
H=224

echo $INPUT_SHAPE

#TRT FP32
python onnx_transform.py --weights="weights/yolov8n-cls_fp32.pt" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W
python build_trt.py --weights="weights/yolov8n-cls_fp32.onnx"  --fp32 --input_shape $BATCH_SIZE $C $H $W

#TRT FP16
python onnx_transform.py --weights="weights/yolov8n-cls_fp16.pt" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W
python build_trt.py --weights="weights/yolov8n-cls_fp16.onnx"  --fp16 --input_shape $BATCH_SIZE $C $H $W

#TRT INT8
rm -r outputs/cache
python onnx_transform.py --weights="weights/yolov8n-cls_int8.pt" --pretrained --network="$NETWORK" --input_shape $BATCH_SIZE $C $H $W
python build_trt.py --weights="weights/yolov8n-cls_int8.onnx"  --int8 --input_shape $BATCH_SIZE $C $H $W