import argparse

import subprocess
import re

import torch
from torchsummary import summary

def main(opt):
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available.')

    global best_prec1, device
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    if opt.verbose:
        if not opt.trt:
            if opt.network == "mobilenet":
                model = torch.hub.load('pytorch/vision:v0.15.2', "mobilenet_v2", weights=f'MobileNet_V2_Weights.DEFAULT')
            elif "resnet" in opt.network:
                model = torch.hub.load('pytorch/vision:v0.15.2', opt.network, weights=f'ResNet{opt.network[6:]}_Weights.DEFAULT')
            elif "yolo" in opt.network:
                #from ultralytics import YOLO
                #YOLOv8 = YOLO(opt.weights)
                from ultralytics.nn.autobackend import AutoBackend
                YOLOv8 = AutoBackend(opt.weights, device=device) #, dnn=False, fp16=False)
                model = YOLOv8.model.fuse()
            else:
                print("Red no reconocida.")
            model.to(device)
            print("================================ PYTORCH SUMM ===============================================\n")
            summary(model, (3,224,224)) 
        else:
            # Ejecutar el comando
            print("================================ ENGINE SUMM ===============================================\n")
            print(format_output(subprocess.Popen("polygraphy inspect model {} --show layers".format(opt.engine), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode('utf-8')))
            print("================================ ONNNX SUMM ===============================================\n")
            print(format_output(subprocess.Popen("polygraphy inspect model {} --show layers --display-as=trt".format((opt.engine).replace('.engine', '.onnx')), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode('utf-8')))
            # Obtener la salida y el error, si lo hay
    if opt.trt:
        print("number of layers: ", get_layers(opt))
        print("number of parameters: ", get_parametros(opt))
    return

def get_layers(opt):
    if opt.trt:
        cmd = f"polygraphy inspect model {opt.engine}"
    else:
        cmd = f"polygraphy inspect model {(opt.engine).replace('.engine', '.onnx')} --display-as=trt"

    # Ejecuta el comando y captura la salida
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"---- (\d+) Layer\(s\) ----", output)
    # Extrae el número de capas si se encuentra el patrón
    if match:
        num_layers = int(match.group(1))
        return num_layers
    else:
        print("No se encontró el número de capas")
        return 0

def get_parametros(opt):
    if opt.trt:
        cmd = f"python post_processing/param_counter.py --engine ../{opt.engine}"
    else:
        cmd = f"onnx_opcounter {(opt.engine).replace('.engine', '.onnx')}"

    # Ejecuta el comando y captura la salida
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    # Decodifica la salida a texto
    output = stdout.decode()

    # Usa una expresión regular para encontrar el número de capas
    match = re.search(r"Number of parameters in the model: (\d+)", output)
    if match:
        num_parameters = int(match.group(1))
        return num_parameters
    else:
        print("No se encontró el número de parametros")
        return 0

def format_output(output):
    # Encontrar todas las capas y sus detalles
    layer_pattern = re.compile(r"Layer (\d+).*?\[Op: (.*?)\](.*?)(?=Layer \d+|\Z)", re.DOTALL)
    layers = layer_pattern.findall(output)

    shape_pattern = re.compile(r"shape=\((.*?)\)")
    
    formatted_output = "| Layer (type) | Output Shape |\n| ---------------|-----------------|\n"
    for layer_num, layer_type, layer_info in layers:
        # Encontrar todos los shapes en la información de la capa
        shapes = shape_pattern.findall(layer_info)
        if shapes:
            # Tomar el último shape
            last_shape = shapes[-1]
            # Limpieza de la forma para eliminar caracteres innecesarios
            last_shape_cleaned = re.sub(r"[^0-9,]", "", last_shape)
            formatted_output += f"| {layer_type} - {int(layer_num) + 1} | {last_shape_cleaned} |\n"

    return formatted_output

# Usa esta función con tu output como argumento
# Por ejemplo: formatted = format_output(your_output_string)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'weights/best.engine', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('--engine', default = 'weights/best.engine', type=str, help='directorio y nombre del engine generado por build_trt.py')
    parser.add_argument('-trt','--trt', action='store_true',help='evaluate model on validation set al optimizar con tensorrt')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    parser.add_argument('--verbose', action='store_true',help='muestra resumen del modelo')
   
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
