import os
import shutil

# Dirección de la carpeta que contiene las imágenes
dir_path = 'dataset_val/val/'

# Lista de todos los archivos en la carpeta
all_files = os.listdir(dir_path)

# Iterar sobre cada archivo
for file_name in all_files:
    # Verificar si el archivo sigue el patrón deseado
    if file_name.startswith("ILSVRC2012_val_") and file_name.endswith(".JPEG"):
        # Extraer el número de label (nX) del nombre del archivo
        label = file_name.split('_')[-1].split('.')[0]

        # Crear la carpeta para el label si no existe
        label_folder_path = os.path.join(dir_path, label)
        if not os.path.exists(label_folder_path):
            os.makedirs(label_folder_path)

        # Mover el archivo a la carpeta correspondiente
        src_path = os.path.join(dir_path, file_name)
        dest_path = os.path.join(label_folder_path, file_name)
        shutil.move(src_path, dest_path)

print("¡Proceso completado!")