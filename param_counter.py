import tensorrt as trt
import os
import torch 
from pathlib import Path
import json

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

#cargamos el engine
current_directory = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(current_directory,'weights/yolov8n-cls.engine')
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, namespace='')
with trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
    
#seguimos los pasos segun la doc de nvidia trt
inspector = engine.create_engine_inspector()

# Crear un inspector de engine

inspector = engine.create_engine_inspector()


total_weights_count = 0

for layer_index in range(engine.num_layers):
    layer_info_json = inspector.get_layer_information(layer_index, trt.LayerInformationFormat.JSON)
    
    # Convertir la información de la capa a un diccionario
    layer_info = json.loads(layer_info_json)

    # Verificar si la capa tiene la clave 'Weights' y sumar el 'Count'
    if 'Weights' in layer_info:
        total_weights_count += layer_info['Weights']['Count']

print(f"Total de 'weights count' en el engine: {total_weights_count}")