import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes los siguientes tiempos promedio de inferencia (en segundos) para cada batch size
# (Estos son solo datos de ejemplo, reemplácelos con los tuyos)
times_vanilla = np.array([0.0084, 0.141, 0.3, 0.6, 1.1])
times_trt_fp32 = np.array([0.0144, 0.078, 0.1, 0.3, 0.6])
times_trt_fp16 = np.array([0.0067, 0.031, 0.057, 0.1, 0.2])

batch_sizes = np.array([1, 32, 64, 128, 256])

# Convertir tiempos de inferencia en throughput (throughput = batch_size / tiempo)
throughput_vanilla = batch_sizes / times_vanilla
throughput_trt_fp32 = batch_sizes / times_trt_fp32
throughput_trt_fp16 = batch_sizes / times_trt_fp16

# Graficar
plt.figure(figsize=(10, 6))
plt.plot(batch_sizes, throughput_vanilla, '-o', label='vanilla')
plt.plot(batch_sizes, throughput_trt_fp32, '-o', label='TRT fp32')
plt.plot(batch_sizes, throughput_trt_fp16, '-o', label='TRT fp16')
plt.xlabel('Batch Size')
plt.ylabel('Inference Throughput (samples/sec)')
plt.title('Inference Throughput vs Batch Size')
plt.legend()
plt.grid(True)
plt.savefig('inference_throughput_vs_batch_size.png')
plt.show()
