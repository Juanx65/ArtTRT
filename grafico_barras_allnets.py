import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que tus datos están en el siguiente formato:
# {"network_name": {"vanilla": latency, "TRT_fp32": latency, "TRT_fp16": latency, "TRT_int8": latency}, ...}

data = {
    "resnet18": {"Vanilla": 2.0, "TRT_fp32": 1.4, "TRT_fp16": 0.8 , "TRT_int8": 0.7},
    "resnet34": {"Vanilla": 3.1, "TRT_fp32": 2.1 , "TRT_fp16": 1.0 , "TRT_int8": 0.8},
    "resnet50": {"Vanilla": 3.7, "TRT_fp32": 2.2  , "TRT_fp16": 1.0 , "TRT_int8": 0.9},
    "resnet101": {"Vanilla": 5.8, "TRT_fp32": 3.8  , "TRT_fp16": 1.7 , "TRT_int8": 1.2 },
    "resnet152": {"Vanilla": 8.1 , "TRT_fp32": 5.5   , "TRT_fp16": 2.2  , "TRT_int8": 1.5 },
    "mobilenet": {"Vanilla": 2.6, "TRT_fp32": 0.9, "TRT_fp16": 0.8 , "TRT_int8": 0.7},
    "yolon": {"Vanilla": 1.8, "TRT_fp32": 0.8, "TRT_fp16": 0.8, "TRT_int8": 0.7},
    "yolox": {"Vanilla": 5.8, "TRT_fp32": 4.1, "TRT_fp16": 1.7 , "TRT_int8": 1.3}
    #... [Similar data for other networks]
}

# Configuraciones del gráfico
barWidth = 0.2
labels = list(data.keys())
vanilla = [values["Vanilla"] for values in data.values()]
TRT_fp32 = [values["TRT_fp32"] for values in data.values()]
TRT_fp16 = [values["TRT_fp16"] for values in data.values()]
TRT_int8 = [values["TRT_int8"] for values in data.values()]

# Configurando la posición de las barras en X
r1 = np.arange(len(vanilla))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

# Creando el gráfico de barras
plt.bar(r1, vanilla,  width=barWidth, edgecolor='white', label='vanilla')
plt.bar(r2, TRT_fp32,  width=barWidth, edgecolor='white', label='TRT_fp32')
plt.bar(r3, TRT_fp16,  width=barWidth, edgecolor='white', label='TRT_fp16')
plt.bar(r4, TRT_int8,  width=barWidth, edgecolor='white', label='TRT_int8')


# Añadiendo etiquetas al gráfico
plt.xlabel('networks', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(vanilla))], labels, rotation=20)
plt.ylabel('Latency ms' , fontweight='bold')

# Creando la leyenda y mostrando el gráfico
plt.legend()
plt.title('Latencia promedio por Red para Batch Size 1')
plt.grid()
plt.savefig('inference_time_bar_all.png')
plt.show()
