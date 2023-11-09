#!/bin/bash

# remeber to do a chmod +x main_all.sh befor runing with ./main_all.sh
BATCH_SIZE=1

NETWORK="resnet18"

#VANILLA
python3 main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" --less
echo "VANILLA ENDS"

#TRT FP32
python3 main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp32.engine' --less
echo "TRT FP32 ENDS"

#TRT FP16
python3 main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp16.engine' --less
echo "TRT FP16 ENDS"