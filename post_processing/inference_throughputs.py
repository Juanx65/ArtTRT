import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes los siguientes tiempos promedio de inferencia (en segundos) para cada batch size
# (Estos son solo datos de ejemplo, reemplácelos con los tuyos)
times_vanilla = np.array([1.9, 26, 48.1, 89.2, 177.5])
times_trt_fp32 = np.array([1.3, 19.1, 34.8, 66.9, 133.2])
times_trt_fp16 = np.array([0.8 , 10.7 , 20.3,38.3, 75.7])
times_trt_int8 = np.array([0.7 , 8.3, 15.8, 30.3, 0])

batch_sizes = np.array([1, 32, 64, 128, 256])

# Convertir tiempos de inferencia en throughput (throughput = batch_size / tiempo)
throughput_vanilla = batch_sizes / times_vanilla 
throughput_trt_fp32 = batch_sizes / times_trt_fp32
throughput_trt_fp16 = batch_sizes / times_trt_fp16
throughput_trt_int8 = batch_sizes / times_trt_int8
# Graficar
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, throughput_vanilla* 1000, '-o', label='vanilla')
plt.plot(batch_sizes, throughput_trt_fp32* 1000, '-o', label='TRT fp32')
plt.plot(batch_sizes, throughput_trt_fp16* 1000, '-o', label='TRT fp16')
plt.plot(batch_sizes, throughput_trt_int8* 1000, '-o', label='TRT int8')
plt.xlabel('Batch Size')
plt.ylabel('Inference Throughput (samples/sec)')
plt.title('Inference Throughput vs Batch Size')
plt.legend()
plt.grid(True)
plt.savefig('inference_throughput_vs_batch_size_resnet18.png')
plt.show()
