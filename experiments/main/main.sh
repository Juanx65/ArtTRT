# Remember to run chmod +x experiments/main.sh before executing ./experiments/main.sh

BATCH_SIZE=$1  # Batch size to use in the experiment, from 1 to 128
NETWORK=$2  # Neural network to use in the experiment, options: mobilenet, resnet18, resnet152
BUILD=$3  # If this input is "build", it will generate new engines; otherwise, it will try to use saved ones
DATASET_PATH=$4  # Validation dataset path, e.g., "datasets/dataset_val/val"; if none, the script will execute a test with random inputs
PROFILE=$5 # if u wanna profile write "--profile" else, leave it al blanck or " "
POWER_MODE=$6 #if you profile on a specific power mode, specify it for the name of the log

C=3  # Number of input channels
W=224  # Input width
H=224  # Input height

# Set the memory threshold (in kilobytes)
MEM_THRESHOLD=102400  # 100MB

# Function to run the Python script
execute() {
    local script=$1
    local python_pid

    # Run the Python script in the background and get its PID

    sudo $script &
    python_pid=$!
    
    #echo "Starting $script with PID $python_pid"

    # Monitor the memory usage of the process
    while true; do
        # Check if the process has finished
        if ! kill -0 $python_pid 2>/dev/null; then
            #echo "$script with PID $python_pid has finished"
            break
        fi

        # Get the memory usage of the process
        mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}') 
        #echo "Available memory: $mem_avail"

        if [ "$mem_avail" -lt "$MEM_THRESHOLD" ]; then
            echo "Memory exceeded ($mem_avail KB available) by $script, terminating PID $python_pid"
            kill -9 $python_pid
            break
        fi
        sleep .1  # Wait for 0.1s before the next check
    done
}

execute_build() {
    local script=$1
    local python_pid

    # Ignore outputs; remove '> /dev/null 2>&1 &' to see output
    sudo $script > /dev/null 2>&1 &
    python_pid=$!

    # Monitor the memory usage of the process
    while true; do
        # Check if the process has finished
        if ! kill -0 $python_pid 2>/dev/null; then
            #echo "$script with PID $python_pid has finished"
            break
        fi

        # Get the memory usage of the process
        mem_avail=$(grep MemAvailable /proc/meminfo | awk '{print $2}') 
        #echo "Available memory: $mem_avail"

        if [ "$mem_avail" -lt "$MEM_THRESHOLD" ]; then
            echo "Memory exceeded ($mem_avail KB available) by $script, terminating PID $python_pid"
            kill -9 $python_pid
            break
        fi
        sleep .1
    done
}

# BUILDS
if [ "$BUILD" = "build" ]; then
    if [ "$BATCH_SIZE" = 1 ]; then
        execute_build "./onnx_transform.py --weights weights/best.pth --pretrained --network $NETWORK --input_shape 1 $C $H $W"
        execute_build "./build_trt.py --weights weights/best.onnx --fp32 --input_shape 1 $C $H $W --engine_name best_fp32.engine"
        execute_build "./build_trt.py --weights weights/best.onnx --fp16 --input_shape 1 $C $H $W --engine_name best_fp16.engine"
        rm -r outputs/cache > /dev/null 2>&1
        execute_build "./build_trt.py --weights weights/best.onnx --int8 --input_shape 1 $C $H $W --engine_name best_int8.engine"
    else  # If the batch size is not 1, it will build a dynamic batch size denoted as -1
        execute_build "./onnx_transform.py --weights weights/best.pth --pretrained --network $NETWORK --input_shape -1 $C $H $W"
        execute_build "./build_trt.py --weights weights/best.onnx --fp32 --input_shape -1 $C $H $W --engine_name best_fp32.engine"
        execute_build "./build_trt.py --weights weights/best.onnx --fp16 --input_shape -1 $C $H $W --engine_name best_fp16.engine"
        rm -r outputs/cache > /dev/null 2>&1
        execute_build "./build_trt.py --weights weights/best.onnx --int8 --input_shape -1 $C $H $W --engine_name best_int8.engine"
    fi
fi

# EXECUTIONS
# Vanilla (BASE MODEL) WE ADD --engine to indicate the ONNX of origin; with this ONNX, we calculate the network parameters
# If you do not specify a dataset, this program will perform an evaluation with random inputs, providing only latency results.
if [ -z "$DATASET_PATH" ] || [ "$DATASET_PATH" == "none" ]; then
    VANILLA="experiments/main/main.py --batch_size $BATCH_SIZE --network $NETWORK --model_version Vanilla --log_dir outputs/log/log_vanilla_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    FP32="experiments/main/main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_fp32.engine --model_version FP32 --log_dir outputs/log/log_fp32_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    FP16="experiments/main/main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_fp16.engine --model_version FP16 --log_dir outputs/log/log_fp16_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    INT8="experiments/main/main.py --batch_size $BATCH_SIZE --network $NETWORK -trt --engine weights/best_int8.engine --model_version INT8 --log_dir outputs/log/log_int8_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
else
    VANILLA="experiments/main/main.py -v --batch_size $BATCH_SIZE --dataset $DATASET_PATH --network $NETWORK --less --engine weights/best.engine --model_version Vanilla --log_dir outputs/log/log_vanilla_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    FP32="experiments/main/main.py -v --batch_size $BATCH_SIZE --dataset $DATASET_PATH --network $NETWORK -trt --engine weights/best_fp32.engine --less --non_verbose --model_version TRT_fp32 --log_dir outputs/log/log_fp32_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    FP16="experiments/main/main.py -v --batch_size $BATCH_SIZE --dataset $DATASET_PATH --network $NETWORK -trt --engine weights/best_fp16.engine --less --non_verbose --model_version TRT_fp16 --log_dir outputs/log/log_fp16_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
    INT8="experiments/main/main.py -v --batch_size $BATCH_SIZE --dataset $DATASET_PATH --network $NETWORK -trt --engine weights/best_int8.engine --less --non_verbose --model_version TRT_int8 --log_dir outputs/log/log_int8_${NETWORK}_bs_${BATCH_SIZE}_${POWER_MODE} $PROFILE"
fi

#sudo rm -r outputs/log > /dev/null 2>&1
#rm post_processing/*.txt > /dev/null 2>&1

# Execute Python scripts sequentially
execute "$VANILLA"
execute "$FP32"
execute "$FP16"
execute "$INT8"