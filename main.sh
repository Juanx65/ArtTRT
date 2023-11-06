#!/bin/bash

# Definir el archivo de salida para el monitoreo de la GPU
GPU_MONITOR_OUTPUT="outputs/gpu_usage/gpu_usage.csv"

# Iniciar nvidia-smi en modo de monitor y escribir la salida a un archivo en segundo plano
nvidia-smi dmon -i 0 -s cmu -d 1 -o TD -f "$GPU_MONITOR_OUTPUT" &

# Guardar el PID del proceso nvidia-smi para poder terminarlo después
NVIDIA_SMI_PID=$!

echo "Iniciando el monitoreo de la GPU con PID $NVIDIA_SMI_PID"

# Ejecutar el script de Python
python main.py -v --batch_size=256 --dataset='datasets/dataset_val/val' --network="resnet50"

# Detener el monitoreo de la GPU una vez que el script de Python haya finalizado
kill $NVIDIA_SMI_PID

echo "Monitoreo de la GPU detenido."

# echo "Procesando el archivo de salida de la GPU..."
python post_processing/gpu_metrics_plotter.py --cvs "$GPU_MONITOR_OUTPUT"