#!/bin/bash
# remeber to do a chmod +x juanjo_test.sh befor runing with ./juanjo_test.sh
BUILD=$1

BATCH_SIZE=1
NX=2

# Función para ejecutar y monitorear un programa de Python
execute_and_monitor() {
    local script=$1
    local output_name=$2
    local tegrastat_pid

    tegrastats --interval 100 > $output_name & #100 ms de sampleo
    python $script
    tegrastats --stop
}

#BUILDS
if [ "$BUILD" = "build" ]; then
    python onnx_transform.py --weights weights/juanjo.pth --input_shape $BATCH_SIZE $NX > /dev/null 2>&1
    python build_trt.py --weights weights/juanjo.onnx  --fp32 --input_shape $BATCH_SIZE $NX --engine_name juanjo_fp32.engine > /dev/null 2>&1
    python build_trt.py --weights weights/juanjo.onnx  --fp16 --input_shape $BATCH_SIZE $NX --engine_name juanjo_fp16.engine > /dev/null 2>&1
fi

#EJECUCIONES
VANILLA="juanjo_experiment.py  -e"
FP32="juanjo_experiment.py  -e -trt --engine weights/juanjo_fp32.engine"
FP16="juanjo_experiment.py  -e -trt --engine weights/juanjo_fp16.engine"
# Ejecutar y monitorear cada script de Python secuencialmente
rm post_processing/*.txt > /dev/null 2>&1
execute_and_monitor "$VANILLA" "post_processing/vanilla.txt"
execute_and_monitor "$FP32" "post_processing/trt_fp32.txt"
execute_and_monitor "$FP16" "post_processing/trt_fp16.txt"
