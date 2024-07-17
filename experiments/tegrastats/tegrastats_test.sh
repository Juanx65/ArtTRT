#!/bin/bash

# test a single net for the tegrastat metrics

BATCH_SIZE=1
NETWORK="resnet18"

C=3
W=224
H=224

#EJECUCIONES
#VANILLA="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK --less --engine weights/best_fp32.engine --model_version Vanilla"
#FP32="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32"
#FP16="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16"
#INT8="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8"

tegrastats --interval 1 --logfile tegrastats_test.txt & #sudo tegrastats si necesitas ver mas metricas
tegrastat_pid=$!
python main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8
kill -9 $tegrastat_pid