import json
import matplotlib.pyplot as plt

# Lista de archivos JSON que contienen los perfiles (actualizar con tus archivos)
json_files = ['log/fp32.json', 'log/fp16.json', 'log/int8.json']

# Función para extraer el uso de memoria del JSON
def extract_memory_usage(file_path):
    timestamps = []
    total_allocated = []
    total_reserved = []

    # Abrir el archivo y procesar el JSON completo
    with open(file_path, 'r') as f:
        data = json.load(f)

    # Verificar si existe la clave "traceEvents"
    if 'traceEvents' in data:
        for entry in data['traceEvents']:
            if (
                isinstance(entry, dict) and
                entry.get("ph") == "i" and
                entry.get("cat") == "cpu_instant_event" and
                entry.get("args", {}).get("Device Id") == 0
            ):
                timestamps.append(entry["ts"])
                # Convertir bytes a megabytes
                total_allocated.append(entry["args"]["Total Allocated"] / 1024**2)  
                total_reserved.append(entry["args"]["Total Reserved"] / 1024**2)  

    return timestamps, total_allocated, total_reserved

# Inicialización del gráfico
plt.figure(figsize=(12, 6))

# Procesar cada archivo JSON
has_data = False  # Variable para verificar si hay datos válidos

for i, json_file in enumerate(json_files):
    timestamps, total_allocated, total_reserved = extract_memory_usage(json_file)
    
    # Convertir timestamps a segundos relativos si hay datos válidos
    if timestamps:
        has_data = True
        base_time = min(timestamps)
        relative_timestamps = [(ts - base_time) / 1e6 for ts in timestamps]  # Convertir a segundos
    
        # Graficar las líneas
        plt.plot(relative_timestamps, total_allocated, label=f'Total Allocated - File {i+1}')
        plt.plot(relative_timestamps, total_reserved, linestyle='--', label=f'Total Reserved - File {i+1}')

# Configuración del gráfico solo si hay datos válidos
if has_data:
    plt.xlabel('Time (seconds)')
    plt.ylabel('Memory (MB)')  # Actualización de la etiqueta
    plt.title('Memory Usage Comparison Across Different JSON Profiles')
    plt.legend()
    plt.grid(True)
    plt.savefig('test.pdf', format='pdf')
    plt.show()
else:
    print("No valid data found in the JSON files to plot.")
