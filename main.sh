#!/bin/bash
BATCH_SIZE=$1 #128
NETWORK=$2 #"resnet18"

C=3
W=224
H=224

# Ajusta el umbral de memoria (en kilobytes)
MEM_THRESHOLD=204800 # 200Mb

# Función para ejecutar y monitorear un programa de Python
execute_and_monitor() {
    local script=$1
    local ignore_output=$2
    local output_name=$3
    local tegrastat_pid
    local pid

    # Ejecutar el script de Python en segundo plano y obtener su PID
    # Verificar si se debe ignorar la salida
    if [ "$ignore_output" = "ignore" ]; then
        python $script > /dev/null 2>&1 &
    else
        if [ "$ignore_output" = "jetson" ]; then
            tegrastats --interval 1 --logfile $output_name & #sudo tegrastats si necesitas ver mas metricas
            tegrastat_pid=$!
        fi
        python $script &
    fi
    pid=$!

    #echo "Iniciando $script con PID $pid"

    # Monitorear el uso de memoria del proceso
    while true; do
        # Revisar si el proceso terminó
        if ! kill -0 $pid 2>/dev/null; then
            #echo "$script con PID $pid terminó"
            #detenemos tegrastats
            if [ "$ignore_output" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            break
        fi

        # Obtener el uso de memoria del proceso
        mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}') 
        #echo "Memoria disp: $mem_avail"

        if [ "$mem_avail" -lt "$MEM_THRESHOLD" ]; then
            echo "Memoria excedida ($mem_avail KB disponibles) por $script, terminando PID $pid"
            if [ "$ignore_output" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            kill -9 $pid
            break
        fi
        sleep .001 # Esperar un segundo antes de la próxima comprobación
    done
}

#BUILDS
ONNX_FP32="onnx_transform.py --weights weights/best_fp32.pth --pretrained --network $NETWORK --input_shape $BATCH_SIZE $C $H $W"
TRT_FP32="build_trt.py --weights weights/best_fp32.onnx  --fp32 --input_shape $BATCH_SIZE $C $H $W"
ONNX_FP16="onnx_transform.py --weights weights/best_fp16.pth --pretrained --network $NETWORK --input_shape $BATCH_SIZE $C $H $W"
TRT_FP16="build_trt.py --weights weights/best_fp16.onnx  --fp16 --input_shape $BATCH_SIZE $C $H $W"
ONNX_INT8="onnx_transform.py --weights weights/best_int8.pth --pretrained --network $NETWORK --input_shape $BATCH_SIZE $C $H $W"
TRT_INT8="build_trt.py --weights weights/best_int8.onnx  --int8 --input_shape $BATCH_SIZE $C $H $W"

#EJECUCIONES
VANILLA="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK --less --engine weights/best_fp32.engine --model_version Vanilla"
FP32="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32"
FP16="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16"
INT8="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8"
# Agrega los demás scripts según sea necesario

# Ejecutar y monitorear cada script de Python secuencialmente
execute_and_monitor "$VANILLA" "jetson" "post_processing/vanilla.txt"

execute_and_monitor "$ONNX_FP32" "ignore"
execute_and_monitor "$TRT_FP32" "ignore"
execute_and_monitor "$FP32" "jetson" "post_processing/trt_fp32.txt"

execute_and_monitor "$ONNX_FP16" "ignore"
execute_and_monitor "$TRT_FP16" "ignore"
execute_and_monitor "$FP16" "jetson" "post_processing/trt_fp16.txt"

rm -r outputs/cache
execute_and_monitor "$ONNX_INT8" "ignore"
execute_and_monitor "$TRT_INT8" "ignore"
execute_and_monitor "$INT8" "jetson" "post_processing/trt_int8.txt"
