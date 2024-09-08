import re
import matplotlib.pyplot as plt

# Función para analizar una línea de estadísticas
# 03-17-2024 18:54:10 RAM 10239/62841MB (lfb 442x4MB) SWAP 1/31421MB (cached 0MB) CPU [0%@1728,0%@1728,0%@1728,0%@1728,0%@1420,0%@1420,0%@1420,0%@1420,off,off,off,off] GR3D_FREQ 0% cpu@37.5C tboard@27C soc2@34.562C tdiode@27.5C soc0@34.531C tj@37.5C soc1@33.656C VDD_GPU_SOC 1592mW/1592mW VDD_CPU_CV -398mW/-398mW VIN_SYS_5V0 2624mW/2624mW VDDQ_VDD2_1V8AO 403mW/403mW
def parse_line(line):
    stats = {}
    parts = line.split()
    stats['time'] = parts[0] + ' ' + parts[1]
    stats['RAM'] = int(parts[3].split('/')[0])
    stats['maxRAM'] = int(parts[3].split('/')[1].split('M')[0])
    stats['SWAP'] = int(parts[7].split('/')[0])
    stats['maxSWAP'] = int(parts[7].split('/')[1].split('M')[0])
    stats['CPU'] = parts[11].strip('[]').split(',')
    stats['GR3D_FREQ'] = int(parts[13].split('%')[0])
    # Agrega aquí más métricas según sea necesario.
    return stats

# Leer datos de los archivos y almacenar en diccionarios
def read_data(file_path):
    data = {'time': [], 'RAM': [], 'SWAP': [], 'GR3D_FREQ': [], 'CPU': [], 'CPU_FREQ': [], 'maxRAM': 0, 'maxSWAP': 0}
    with open(file_path, 'r') as file:
        first_line = True
        time_ms = 0  # Inicializa el contador de tiempo en 0
        for line in file:
            if line.strip():
                stats = parse_line(line)
                if first_line:
                    # Extrae maxRAM y maxSWAP solo de la primera línea
                    data['maxRAM'] = stats['maxRAM']
                    data['maxSWAP'] = stats['maxSWAP']
                    first_line = False
                 # Calcula el promedio del uso de CPU y la frecuencia
                cpu_usage_sum = 0
                cpu_freq_sum = 0
                cpu_count = 0
                for cpu in stats['CPU']:
                    if cpu != 'off':
                        usage, freq = cpu.split('@')
                        cpu_usage_sum += int(usage[:-1])  # Elimina el signo % y convierte a int
                        cpu_freq_sum += int(freq)
                        cpu_count += 1
                
                # Solo añade valores si hay CPUs activos
                if cpu_count > 0:
                    avg_cpu_usage = cpu_usage_sum / cpu_count
                    avg_cpu_freq = cpu_freq_sum / cpu_count
                else:
                    avg_cpu_usage = 0
                    avg_cpu_freq = 0

                data['CPU'].append(avg_cpu_usage)
                data['CPU_FREQ'].append(avg_cpu_freq)
                data['time'].append(time_ms)
                data['RAM'].append(stats['RAM'])
                data['SWAP'].append(stats['SWAP'])
                data['GR3D_FREQ'].append(stats['GR3D_FREQ'])
                # Incrementa el contador de tiempo en 1 milisegundo para el siguiente registro
                time_ms += 1 # estamos probando tegrastats a una taza de sampleo de 1 ms
    return data

def plot_data(metrics, labels, title):
    # Extraer maxRAM y maxSWAP de los datos (asumiendo que son constantes)
    maxRAM = metrics[0]['maxRAM']
    maxSWAP = metrics[0]['maxSWAP']

    # Graficar CPU
    plt.figure(figsize=(10, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(metric['time'], metric['CPU'], label=f'CPU {label}')
        max_ram = max(metric['CPU'])
        avg_ram = sum(metric['CPU']) / len(metric['CPU'])
        plt.text(0.5, 0.9-(index*0.05), f'Max CPU {label}: {max_ram} %, Avg CPU {label}: {avg_ram:.2f} %', horizontalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{title} - CPU %')
    plt.xlabel('Time ms')
    plt.ylabel('CPU Avg Saturation (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title}_CPU.pdf', bbox_inches='tight')
    plt.close()
    
    # Graficar RAM
    plt.figure(figsize=(10, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(metric['time'], metric['RAM'], label=f'RAM {label}')
        max_ram = max(metric['RAM'])
        avg_ram = sum(metric['RAM']) / len(metric['RAM'])
        plt.text(0.5, 0.9-(index*0.05), f'Max RAM {label}: {max_ram} Mb, Avg RAM {label}: {avg_ram:.2f} Mb', horizontalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{title} - RAM Usage')
    plt.xlabel('Time ms')
    plt.ylabel('RAM Usage Mb')
    plt.ylim(0, maxRAM)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title}_RAM.pdf', bbox_inches='tight')
    plt.close()

    # Graficar SWAP
    plt.figure(figsize=(10, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(metric['time'], metric['SWAP'], label=f'SWAP {label}')
        max_swap = max(metric['SWAP'])
        avg_swap = sum(metric['SWAP']) / len(metric['SWAP'])
        plt.text(0.5, 0.9-(index*0.05), f'Max SWAP {label}: {max_swap} Mb, Avg SWAP {label}: {avg_swap:.2f} Mb', horizontalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{title} - SWAP Usage')
    plt.xlabel('Time ms')
    plt.ylabel('SWAP Usage Mb')
    plt.ylim(0, maxSWAP)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title}_SWAP.pdf', bbox_inches='tight')
    plt.close()

    # Graficar GR3D_FREQ
    plt.figure(figsize=(10, 5))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        plt.plot(metric['time'], metric['GR3D_FREQ'], label=f'GR3D_FREQ {label}')
        max_gr3d_freq = max(metric['GR3D_FREQ'])
        avg_gr3d_freq = sum(metric['GR3D_FREQ']) / len(metric['GR3D_FREQ'])
        plt.text(0.5, 0.9-(index*0.05), f'Max GR3D_FREQ {label}: {max_gr3d_freq}%, Avg GR3D_FREQ {label}: {avg_gr3d_freq:.2f}%', horizontalalignment='center', transform=plt.gca().transAxes)
    plt.title(f'{title} - GR3D_FREQ Saturation %')
    plt.xlabel('Time ms')
    plt.ylabel('GPU Saturation (%)')
    plt.ylim(0, 100)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{title}_GR3D_FREQ.pdf', bbox_inches='tight')
    plt.close()


# Leer datos de cada optimización
data_vanilla = read_data('outputs/tegrastats_log/vanilla_mobilenet_bs_1_PM0.txt')
data_trt_fp32 = read_data('outputs/tegrastats_log/fp32_mobilenet_bs_1_PM0.txt')
data_trt_fp16 = read_data('outputs/tegrastats_log/fp16_mobilenet_bs_1_PM0.txt')
data_trt_int8 = read_data('outputs/tegrastats_log/int8_mobilenet_bs_1_PM0.txt')

# Graficar los datos
plot_data(
    [ data_vanilla, data_trt_fp32, data_trt_fp16,data_trt_int8],
    [ 'Vanilla', 'TRT fp32', 'TRT fp16', 'TRT int8'],
    'MetricComparisonOveTrime'
)
#[ data_vanilla, data_trt_fp32, data_trt_fp16, data_trt_int8],
#[ 'Vanilla', 'TRT fp32', 'TRT fp16', 'TRT INT8'],