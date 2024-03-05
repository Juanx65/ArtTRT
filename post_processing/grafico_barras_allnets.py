import matplotlib.pyplot as plt
import numpy as np

# Suponiendo que tus datos están en el siguiente formato:
# {"network_name": {"vanilla": latency, "TRT_fp32": latency, "TRT_fp16": latency, "TRT_int8": latency}, ...}

data = {
    "resnet18": {"Vanilla": 17.5, "TRT_fp32": 11.6, "TRT_fp16": 6.7 , "TRT_int8": 5.0},
    "resnet34": {"Vanilla": 30.0, "TRT_fp32": 20.0 , "TRT_fp16": 8.1 , "TRT_int8": 6.5},
    "resnet50": {"Vanilla": 39.7, "TRT_fp32": 29.9  , "TRT_fp16": 9.5 , "TRT_int8": 6.6},
    "resnet101": {"Vanilla": 96.2, "TRT_fp32": 76.8  , "TRT_fp16": 25.3 , "TRT_int8": 13.0 },
    "resnet152": {"Vanilla": 96.2, "TRT_fp32": 76.8   , "TRT_fp16": 25.3  , "TRT_int8": 13.0 },
    "mobilenet": {"Vanilla": 13.8, "TRT_fp32": 11.6, "TRT_fp16": 6.2 , "TRT_int8": 4.7}
    #"yolon": {"Vanilla": 0, "TRT_fp32": 0, "TRT_fp16": 0, "TRT_int8": 0},
    #"yolox": {"Vanilla": 0, "TRT_fp32": 0, "TRT_fp16": 0 , "TRT_int8": 0}
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
plt.savefig('inference_time_bar_all.pdf', format='pdf')
plt.show()