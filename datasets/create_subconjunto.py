import os
import shutil
import random

# Ruta al dataset de validación original de ImageNet
ruta_val_dataset = "dataset_val/val"
# Ruta donde se creará el nuevo subdataset
ruta_subdataset = "subdataset_val/val"

# Crear el directorio para el subdataset
os.makedirs(ruta_subdataset, exist_ok=True)

# Obtener la lista de clases (carpetas) en el dataset de validación
clases = os.listdir(ruta_val_dataset)

# Recorrer cada clase y seleccionar 5 imágenes aleatorias
for clase in clases:
    ruta_clase = os.path.join(ruta_val_dataset, clase)
    imagenes = os.listdir(ruta_clase)
    
    # Seleccionar 5 imágenes aleatorias
    imagenes_seleccionadas = random.sample(imagenes, 5)

    # Crear la carpeta de la clase en el subdataset
    ruta_clase_subdataset = os.path.join(ruta_subdataset, clase)
    os.makedirs(ruta_clase_subdataset, exist_ok=True)

    # Copiar las 5 imágenes seleccionadas a la carpeta correspondiente en el subdataset
    for imagen in imagenes_seleccionadas:
        ruta_origen = os.path.join(ruta_clase, imagen)
        ruta_destino = os.path.join(ruta_clase_subdataset, imagen)
        shutil.copy(ruta_origen, ruta_destino)

print("Subdataset de validación creado con éxito.")
