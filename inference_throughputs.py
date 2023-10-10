import numpy as np
import matplotlib.pyplot as plt

# Supongamos que tienes los siguientes tiempos promedio de inferencia (en segundos) para cada batch size
# (Estos son solo datos de ejemplo, reemplácelos con los tuyos)
times_vanilla = np.array([8.1, 141, 283, 555.1, 1072])
times_trt_fp32 = np.array([5.5, 75.6, 135.2, 141, 592])
times_trt_fp16 = np.array([2.2 , 30.6 , 59.4,108.3, 215.2])
times_trt_int8 = np.array([1.5 , 18.1, 0, 0, 0])

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
plt.savefig('inference_throughput_vs_batch_size.png')
plt.show()
