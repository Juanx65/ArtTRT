#!/bin/bash

# remeber to do a chmod +x main.sh befor runing with ./main.sh
BATCH_SIZE=$1
NETWORK=$2

# remeber to do a chmod +x build.sh befor runing with ./build.sh
./build.sh "$BATCH_SIZE" "$NETWORK" > /dev/null 2>&1 # cambiar  "$BATCH_SIZE" por -1 para construir la red de forma dinamica, no aplica a int8
#./build.sh -1 "$NETWORK" > /dev/null 2>&1 # cambiar  "$BATCH_SIZE" por -1 para construir la red de forma dinamica, no aplica a int8


MONITOR_OUTPUT_VANILLA="outputs/gpu_usage/gpu_usage_vanilla.csv"
MONITOR_OUTPUT_FP32="outputs/gpu_usage/gpu_usage_fp32.csv"
MONITOR_OUTPUT_FP16="outputs/gpu_usage/gpu_usage_fp16.csv"
MONITOR_OUTPUT_INT8="outputs/gpu_usage/gpu_usage_int8.csv"

#VANILLA
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_VANILLA" &
NVIDIA_SMI_PID=$!
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" --less --engine weights/best_fp32.engine --model_version "Vanilla"
kill $NVIDIA_SMI_PID
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_VANILLA" --output outputs/img_readme/"$NETWORK"_bs"$BATCH_SIZE"/gpu_usage_vanilla.png

#TRT FP32
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_FP32" &
NVIDIA_SMI_PID=$!
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp32.engine' --less --non_verbose --model_version "TRT fp32"
kill $NVIDIA_SMI_PID
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_FP32" --output outputs/img_readme/"$NETWORK"_bs"$BATCH_SIZE"/gpu_usage_fp32.png

#TRT FP16
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_FP16" &
NVIDIA_SMI_PID=$!
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp16.engine' --less --non_verbose --model_version "TRT fp16"
kill $NVIDIA_SMI_PID
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_FP16" --output outputs/img_readme/"$NETWORK"_bs"$BATCH_SIZE"/gpu_usage_fp16.png

#TRT INT8
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_INT8" &
NVIDIA_SMI_PID=$!
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_int8.engine' --less --non_verbose --model_version "TRT int8"
kill $NVIDIA_SMI_PID
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_INT8" --output outputs/img_readme/"$NETWORK"_bs"$BATCH_SIZE"/gpu_usage_int8.png