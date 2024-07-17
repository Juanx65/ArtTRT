#!/bin/bash
# remeber to do a chmod +x juanjo_test.sh befor runing with ./juanjo_test.sh
BUILD=$1
L=$2
M=$3
BTT=$4

BATCH_SIZE=1
NX=2 #como una imagen 640x640=409600
NU=1

# Función para ejecutar y monitorear un programa de Python
execute_and_monitor() {
    local script=$1
    local output_name=$2
    local tegrastat_pid

    tegrastats --interval 100 > $output_name & #100 ms de sampleo
    $script
    tegrastats --stop
}
#EJECUCIONES
VANILLA="experiments/MPC/juanjo_experiment.py --save_model -e -nx $NX -nu $NU -L $L -M $M -btt $BTT -bs $BATCH_SIZE --name Vanilla"
FP32="experiments/MPC/juanjo_experiment.py  -e -trt --engine weights/juanjo_fp32.engine -nx $NX -nu $NU -L $L -M $M -btt $BTT -bs $BATCH_SIZE --name TRTfp32"
FP16="experiments/MPC/juanjo_experiment.py  -e -trt --engine weights/juanjo_fp16.engine -nx $NX -nu $NU -L $L -M $M -btt $BTT -bs $BATCH_SIZE --name TRTfp16"
INT8="experiments/MPC/juanjo_experiment.py  -e -trt --engine weights/juanjo_int8.engine -nx $NX -nu $NU -L $L -M $M -btt $BTT -bs $BATCH_SIZE --name TRTint8"

# Ejecutar y monitorear cada script de Python secuencialmente
rm post_processing/*.txt > /dev/null 2>&1
execute_and_monitor "$VANILLA" "post_processing/vanilla.txt"

#BUILDS
if [ "$BUILD" = "build" ]; then
    ./onnx_transform.py --weights weights/juanjo.pth --input_shape $BATCH_SIZE $NX > /dev/null 2>&1
    ./build_trt.py --weights weights/juanjo.onnx  --fp32 --input_shape $BATCH_SIZE $NX --engine_name juanjo_fp32.engine > /dev/null 2>&1
    ./build_trt.py --weights weights/juanjo.onnx  --fp16 --input_shape $BATCH_SIZE $NX --engine_name juanjo_fp16.engine > /dev/null 2>&1
    rm -r outputs/cache > /dev/null 2>&1
    ./build_trt.py --weights weights/juanjo.onnx  --int8 --input_shape $BATCH_SIZE $NX --engine_name juanjo_int8.engine > /dev/null 2>&1
fi
execute_and_monitor "$FP32" "post_processing/trt_fp32.txt"
execute_and_monitor "$FP16" "post_processing/trt_fp16.txt"
execute_and_monitor "$INT8" "post_processing/trt_int8.txt"