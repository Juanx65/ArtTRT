#!/bin/bash
BATCH_SIZE=$1 #128
NETWORK=$2 #"resnet18"
BUILD=$3

C=3
W=224
H=224

# Ajusta el umbral de memoria (en kilobytes)
MEM_THRESHOLD=204800 # 200Mb
# Función para ejecutar y monitorear un programa de Python
execute_and_monitor() {
    local script=$1
    local is_jetson=$2
    local output_name=$3
    local tegrastat_pid
    local python_pid

    # Ejecutar el script de Python en segundo plano y obtener su PID
    # Verificar si se debe ignorar la salida

    if [ "$is_jetson" = "jetson" ]; then
        tegrastats --interval 100 --logfile $output_name & #sudo tegrastats si necesitas ver mas metricas
        tegrastat_pid=$!
    fi
    #python $script &
    sudo -E env PATH=${PATH} /home/juam/Documents/ArtTRT/env/bin/python $script & # sudo, for the profiler
    python_pid=$!

    #echo "Iniciando $script con PID $pid"

    # Monitorear el uso de memoria del proceso
    while true; do
        # Revisar si el proceso terminó
        if ! kill -0 $python_pid 2>/dev/null; then
            #echo "$script con PID $python_pid terminó"
            #detenemos tegrastats
            if [ "$is_jetson" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            break
        fi

        # Obtener el uso de memoria del proceso
        mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}') 
        #echo "Memoria disp: $mem_avail"

        if [ "$mem_avail" -lt "$MEM_THRESHOLD" ]; then
            echo "Memoria excedida ($mem_avail KB disponibles) por $script, terminando PID $pid"
            if [ "$is_jetson" = "jetson" ]; then
                kill -9 $tegrastat_pid
            fi
            kill -9 $python_pid
            break
        fi
        sleep .1 # Esperar 1ms antes de la próxima comprobación
    done
}

#BUILDS
if [ "$BUILD" = "build" ]; then
    python onnx_transform.py --weights weights/best.pth --pretrained --network $NETWORK --input_shape $BATCH_SIZE $C $H $W > /dev/null 2>&1
    python build_trt.py --weights weights/best.onnx  --fp32 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp32.engine > /dev/null 2>&1
    python build_trt.py --weights weights/best.onnx  --fp16 --input_shape $BATCH_SIZE $C $H $W --engine_name best_fp16.engine > /dev/null 2>&1
    rm -r outputs/cache > /dev/null 2>&1
    python build_trt.py --weights weights/best.onnx  --int8 --input_shape $BATCH_SIZE $C $H $W --engine_name best_int8.engine > /dev/null 2>&1
fi

#EJECUCIONES
#VANILLA="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK --less --engine weights/best_fp32.engine --model_version Vanilla"
#FP32="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32"
#FP16="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16"
#INT8="main.py -v --batch_size $BATCH_SIZE --dataset datasets/dataset_val/val --network $NETWORK -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8"
## PARA CORRER VERSIONES QUE QUIERO VER CON NSIGHT
VANILLA="main.py --batch_size $BATCH_SIZE --network $NETWORK --log_dir log/log_vanilla --model_version Vanilla"
FP32="main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_fp32.engine --log_dir log/log_fp32 --model_version FP32"
FP16="main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_fp16.engine --log_dir log/log_fp16 --model_version FP16"
INT8="main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_int8.engine --log_dir log/log_int8 --model_version INT8"

sudo rm -r log > /dev/null 2>&1
rm post_processing/*.txt > /dev/null 2>&1
# Ejecutar y monitorear cada script de Python secuencialmente
execute_and_monitor "$VANILLA" "jetson" "post_processing/vanilla.txt"
execute_and_monitor "$FP32" "jetson" "post_processing/trt_fp32.txt"
execute_and_monitor "$FP16" "jetson" "post_processing/trt_fp16.txt"
execute_and_monitor "$INT8" "jetson" "post_processing/trt_int8.txt"