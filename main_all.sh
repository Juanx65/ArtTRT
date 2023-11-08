#!/bin/bash

# remeber to do a chmod +x main_all.sh befor runing with ./main_all.sh
BATCH_SIZE=1

NETWORK="resnet18"

MONITOR_OUTPUT_VANILLA="outputs/gpu_usage/gpu_usage_vanilla.csv"
MONITOR_OUTPUT_FP32="outputs/gpu_usage/gpu_usage_fp32.csv"
MONITOR_OUTPUT_FP16="outputs/gpu_usage/gpu_usage_fp16.csv"
MONITOR_OUTPUT_INT8="outputs/gpu_usage/gpu_usage_int8.csv"

#VANILLA
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_VANILLA" &
NVIDIA_SMI_PID=$!
echo "VANILLA: Iniciando el monitoreo de la GPU con PID $NVIDIA_SMI_PID"
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" --less
kill $NVIDIA_SMI_PID
echo "VANILLA ENDS"
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_VANILLA"


#TRT FP32
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_FP32" &
NVIDIA_SMI_PID=$!
echo "TRT FP32: Iniciando el monitoreo de la GPU con PID $NVIDIA_SMI_PID"
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp32.engine' --less
kill $NVIDIA_SMI_PID
echo "TRT FP32 ENDS"
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_FP32"

#TRT FP16
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_FP16" &
NVIDIA_SMI_PID=$!
echo "TRT FP16: Iniciando el monitoreo de la GPU con PID $NVIDIA_SMI_PID"
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_fp16.engine' --less
kill $NVIDIA_SMI_PID
echo "TRT FP16 ENDS"
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_FP16"


#TRT INT8
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$MONITOR_OUTPUT_INT8" &
NVIDIA_SMI_PID=$!
echo "TRT INT8: Iniciando el monitoreo de la GPU con PID $NVIDIA_SMI_PID"
python main.py -v --batch_size=$BATCH_SIZE --dataset='datasets/dataset_val/val' --network="$NETWORK" -trt --engine='weights/best_int8.engine' --less
kill $NVIDIA_SMI_PID
echo "TRT INT8 ENDS"
python post_processing/gpu_metrics_plotter.py --csv "$MONITOR_OUTPUT_INT8"