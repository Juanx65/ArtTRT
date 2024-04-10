##
##
##  Este codigo es necesario si requieres usar la nueva funcionalidad en los logs de pytorch profiler, que requieren que exista el campo 'distributedInfo' en los json
##
##
import json
import glob

# Encuentra todos los archivos .json en el directorio actual y subdirectorios de manera recursiva
json_files = glob.glob('../outputs/log/**/*.json', recursive=True)
# Ordena los archivos para asegurarse de que la secuencia de rank sea consistente
json_files.sort()

# Inicializa el rank a 0
rank = 0

# Recorre los archivos, lee su contenido, y añade el identificador
for file_path in json_files:
    # Abre y lee el contenido actual del archivo
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    # Prepara un nuevo objeto JSON con 'distributedInfo' al inicio
    new_data = {"distributedInfo": {"rank": rank}, **data}
    
    # Incrementa el rank para el próximo archivo
    rank += 1
    
    # Escribe el objeto modificado de nuevo al archivo
    with open(file_path, 'w') as file:
        json.dump(new_data, file, indent=4)

print(f"Se han actualizado {len(json_files)} archivos .json, añadiendo 'distributedInfo' con rangos incrementales.")
