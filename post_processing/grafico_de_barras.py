import matplotlib.pyplot as plt
import numpy as np

# Datos de entrada (debes completar estos datos con tus valores)
redes = ['Vanilla', 'TRT fp32', 'TRT fp16', 'TRT int8']

# Suponiendo que la latencia está en milisegundos (ajusta los valores de acuerdo a tus datos)
latencias = {
    'Vanilla': [1.9, 26, 48.1, 89.2, 177.5],  # Ejemplo: [latencia para batch size 1, latencia para batch size 32, ...]
    'TRT fp32': [1.3, 19.1, 34.8, 66.9, 133.2],
    'TRT fp16': [0.8 , 10.7 , 20.3,38.3, 75.7],
    'TRT int8': [0.7 , 8.3, 15.8, 30.3, 0]
}



batch_sizes = [1, 32, 64, 128, 256]
bar_width = 0.2
r = np.arange(len(batch_sizes))

# Creando las barras
for idx, red in enumerate(redes):
    plt.bar(r + idx*bar_width, 10*np.log(latencias[red]), width=bar_width, edgecolor='white', label=red)

# Agregar detalles al gráfico
plt.xlabel('Batch Size', fontweight='bold')
plt.ylabel('Latencia promedio (10log ms)', fontweight='bold')
plt.xticks([r + bar_width for r in range(len(batch_sizes))], batch_sizes)
plt.legend()
plt.title('Latencia promedio por Red y Batch Size')

plt.grid()
# Mostrar gráfico
plt.tight_layout()
plt.savefig('inference_time_bar_resnet18.png')
plt.show()
