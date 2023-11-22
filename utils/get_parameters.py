import argparse

import subprocess
import re

def main(opt):
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

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='datasets/dataset_val/val', help='path to dataset')
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size to train')
    parser.add_argument('--weights', default = 'weights/best.engine', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('--engine', default = 'weights/best.engine', type=str, help='directorio y nombre del engine generado por build_trt.py')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
    parser.add_argument('-m','--pin_memmory', action='store_true',help='use pin memmory')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',help='print frequency (default: 10)')
    parser.add_argument('-trt','--trt', action='store_true',help='evaluate model on validation set al optimizar con tensorrt')
    parser.add_argument('-n','--network', default='resnet18',help='name of the pretrained model to use')
    parser.add_argument('-v','--validate', action='store_true',help='validate with validation data')
    parser.add_argument('-c','--compare', action='store_true',help='compare the results of the vanilla model with the trt model using random generated inputs')
    parser.add_argument('-rtol','--rtol', default=1e-3,type=float, help='relative tolerance for the numpy.isclose() function')
    parser.add_argument('-vd','--val_dataset', action='store_true',help='compare the results of the vanilla model with the trt model using the validation dataset as inputs')
    parser.add_argument('--profile', action='store_true',help='profiles the validation run with torch profiler')
    parser.add_argument('--compare_3', action='store_true',help='compare the results of the vanilla model with the trt model using random generated inputs')
    parser.add_argument('--less', action='store_true',help='print less information')
    parser.add_argument('--non_verbose', action='store_true',help='no table header and no gpu information')
    parser.add_argument('--model_version', default='Vanilla',help='model name in the table output (validation): Vanilla, TRT fp32, TRT fp16 TRT int8')
   
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
