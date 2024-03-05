import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes los siguientes tiempos promedio de inferencia (en segundos) para cada batch size
# (Estos son solo datos de ejemplo, reemplácelos con los tuyos)
times_vanilla = np.array([1.7, 24.1, 45.7,88.7 , 177.0])
times_trt_fp32 = np.array([1.3, 17.7, 34.0,66.0, 131.7])
times_trt_fp16 = np.array([0.8 , 10.4 , 19.8,37.7, 74.7])
times_trt_int8 = np.array([0.6 , 8.2, 15.7, 30.3, 0])

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
plt.savefig('inference_throughput_vs_batch_size_resnet18.pdf', format='pdf')
plt.show()
