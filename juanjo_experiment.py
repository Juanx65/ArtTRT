import argparse
import numpy as np
import torch
import time
import os
from torchinfo import summary

from utils.engine import TRTModule
from utils.models.CustomNets import JuanjoNet

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')
global device
device = torch.device("cuda:0" if train_on_gpu else "cpu")

def main(opt):
    # Crear una instancia de la red
    #nx, M, nu, L = entradas, neuronas x capa, salidas, Capas

    net = JuanjoNet(opt.nx, opt.M, opt.nu, opt.L)
    net.inicializar_pesos()
    net.to(device)

    if opt.save_model:
        torch.save(net, opt.weights)

    if opt.trt:
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.engine)

        Engine = TRTModule(engine_path, device)
        Engine.set_desired(['outputs'])
        if opt.evaluate:
            evaluate(opt, Engine)
    elif opt.evaluate:
        evaluate(opt, net)

def evaluate(opt, model):

    if opt.info_model:
        summary(model, (opt.batch_size, opt.nx))

    num_batches = opt.batches_to_test
    total_time = 0
    max_time = 0 # Inicializa la variable para el tiempo máximo
    
    np_seed = 42
    np.random.seed(np_seed)
    inputs = np.random.rand(num_batches, opt.nx)

    for i in range(num_batches):

        input = torch.from_numpy(inputs[i]).float() 

        start_time = time.perf_counter_ns() /1000000
        input = input.to(device)
        with torch.no_grad():
            output = model(input)
            torch.cuda.synchronize()  # Asegura que todas las operaciones en la GPU se han completado
            output = output.cpu()
        end_time = time.perf_counter_ns() /1000000

        cycle_time = end_time - start_time
        total_time += cycle_time

        if cycle_time > max_time:  # Comprueba si el tiempo del ciclo actual es el máximo hasta ahora
            max_time = cycle_time

    average_time = total_time / num_batches
    print(f"{opt.name}: avg {average_time} | max {max_time} ms")
    return


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default = 'weights/juanjo.pth', type=str, help='path to the pth weight file')
    parser.add_argument('--engine', default = 'weights/juanjo.engine', type=str, help='path to the engine weight file')
    parser.add_argument('--name', default = 'Vanilla', type=str, help='Name of the test')
    parser.add_argument('-nx', '--nx', default = 2, type=int, help='input')
    parser.add_argument('-nu', '--nu', default = 1, type=int, help='outputs')
    parser.add_argument('-L', '--L', default = 3, type=int, help='# Capas')
    parser.add_argument('-M', '--M', default = 5, type=int, help='# Neuronas por capa')
    parser.add_argument('-bs', '--batch_size', default = 1, type=int, help='batch size')
    parser.add_argument('-btt', '--batches_to_test', default = 10000, type=int, help='batches to test')
    parser.add_argument('--save_model', action = 'store_true', help='guarda el modelo Vanilla en --weights')
    parser.add_argument('--info_model', action = 'store_true', help='muestra un resumen de la red segun torchinfo')
    parser.add_argument('-e','--evaluate', action = 'store_true', help='evaluar el modelo')
    parser.add_argument('-trt', '--trt', action = 'store_true', help='el modelo es un engine de tensorrt')
    
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    start = time.perf_counter_ns() /1000000
    main(opt)
    end = time.perf_counter_ns() /1000000
    print(end-start)
