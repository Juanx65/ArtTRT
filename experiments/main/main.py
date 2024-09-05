#!/usr/bin/env -S bash -c '"`dirname $(dirname $(dirname $0))`/env/bin/python" "$0" "$@"'

import argparse
import sys
import os
import time
# Añade la ruta al directorio que contiene el módulo utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np

import torch
import torch.nn as nn
import torch.utils.data


from utils.data_loader import val_data_loader
from utils.helper import AverageMeter, accuracy

from torch.profiler import profile, ProfilerActivity,schedule, tensorboard_trace_handler, _KinetoProfile
#from hta.trace_analysis import TraceAnalysis
#from torchsummary import summary

import subprocess
import re

import scipy.stats as stats

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

best_prec1 = 0.0

def main(opt):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    base_directory = os.path.abspath(os.path.join(current_directory, '../../')) #if en experiments/main/main.py
    
    train_on_gpu = torch.cuda.is_available()
    if not opt.non_verbose:
        if not train_on_gpu:
            print('CUDA is not available.')
        else:
            print('CUDA is available.')

    global best_prec1, device
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    if opt.trt:
        from utils.engine import TRTModule #if not done here, unable to train
        engine_path = os.path.join(base_directory,opt.engine)
        Engine = TRTModule(engine_path,device)
        Engine.set_desired(['outputs'])
        if not opt.compare:
            model = Engine
        else:
            Engine.to(device)

    if opt.compare or not opt.trt:
        if opt.network == "mobilenet":
            model = torch.hub.load('pytorch/vision:v0.15.2', "mobilenet_v2", weights=f'MobileNet_V2_Weights.DEFAULT')
        elif "resnet" in opt.network:
            model = torch.hub.load('pytorch/vision:v0.15.2', opt.network, weights=f'ResNet{opt.network[6:]}_Weights.DEFAULT')
        elif "yolo" in opt.network:
            from ultralytics import YOLO
            YOLOv8 = YOLO(os.path.join(base_directory,opt.weights))
            model = YOLOv8.model.fuse()
        else:
            print("Red no reconocida.")
    model.to(device)

    if opt.validate:
        if "yolo" in  opt.network:
            val_loader = val_data_loader(os.path.join(base_directory,opt.dataset), opt.batch_size, opt.workers, opt.pin_memmory, do_normalize=False)
        else:
            val_loader = val_data_loader(os.path.join(base_directory,opt.dataset), opt.batch_size, opt.workers, opt.pin_memmory)
        criterion = nn.CrossEntropyLoss().to(device)
        validate(opt, val_loader, model, criterion, opt.print_freq,opt.batch_size)

    elif opt.compare:
        if "yolo" in  opt.network:
            val_loader = val_data_loader(os.path.join(base_directory,opt.dataset), opt.batch_size, opt.workers, opt.pin_memmory, do_normalize=False)
        else:
            val_loader = val_data_loader(os.path.join(base_directory,opt.dataset), opt.batch_size, opt.workers, opt.pin_memmory)
        #compare the output values of base model and engine using the closeness metric 
        compare(val_loader, model, Engine, opt.batch_size, opt.rtol)

    else:
        # default evaluation with random numbers, no input dataset
        evaluate(opt, model)
    return

def compare(val_loader, model, Engine, batch_size, rtol=1e-3):
    from tabulate import tabulate
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    top_n = 10
    disagreements = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    mae_errors = np.zeros(top_n,dtype=np.float64)

    num_batches = len(val_loader)

    closeness_count = 0
    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break
        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_trt = Engine(input)

        close_values = torch.isclose(output_vanilla, output_trt, rtol=rtol).sum().item()
        closeness_count += close_values/1000  # equal elements / number of elements
            
         # Get top n classes and their scores for each model
        top_scores_vanilla, top_indices_vanilla = torch.topk(output_vanilla, top_n)
        top_scores_trt, top_indices_trt = torch.topk(output_trt, top_n)

        # Compare the top classes and their scores for each position from top1 to top_n
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_trt[0])):
            if idx_v.item() != idx_t.item():
                disagreements[j] += 1
            
            # Get scores of the top class of Engine (trt) in both models
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_trt[0, idx_t].item()

            # MEAN Absolute Error differences
            mae_errors[j] += np.abs(score_v - score_t)

        #Tabla que muestre los top n scores de cada salida
        table = []
        headers = ['Rank', 'Vanilla Score', 'Vanilla Label', 'TRT Score', 'TRT Label']
        for j in range(top_n):
            row = [
                j + 1,
                top_scores_vanilla[0][j].item(),
                top_indices_vanilla[0][j].item(),
                top_scores_trt[0][j].item(),
                top_indices_trt[0][j].item()
            ]
            table.append(row)
        
        print(tabulate(table, headers=headers)) 

    # Get average MSE for each position
    mae_errors = [error / num_batches for error in mae_errors]

    # Convert disagreements to percentages
    total_images = num_batches * batch_size
    disagreement_percentages = [(disagreement / total_images) * 100 for disagreement in disagreements]

    print("Disagreements (in percentage) for each rank class:")
    for j, percentage in enumerate(disagreement_percentages, 1):
        print(f"Top {j}: {percentage:.4f}% disagreements")
    print("Mean Absolute Error for each rank possition:")
    for j, error in enumerate(mae_errors, 1):
        print(f"Top {j}: {error:.8f} MAE")

    total_values = num_batches * batch_size
    closeness_percentage = (closeness_count / total_values) * 100
    print(f"The avg percentage of equal elements (using torch.isclose) for class is {closeness_percentage:.4f} %")
    return

def evaluate(opt, model):
    nun_batches = 12
    torch.manual_seed(42)
    inputs= torch.rand(nun_batches,opt.batch_size, 3, 224, 224) # generamos un input random [0,1)

    #analyzer = TraceAnalysis(trace_dir=opt.log_dir)

    #falta probar con _KinetoProfiler
    if opt.profile:
        tracing_schedule = schedule(skip_first=0, wait=0, warmup=2, active=10, repeat=1)
        #trace_handler = tensorboard_trace_handler(dir_name=opt.log_dir)
        trace_handler = tensorboard_trace_handler(dir_name=os.path.abspath(opt.log_dir))
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = tracing_schedule,
            on_trace_ready = trace_handler,
            profile_memory = True,
            record_shapes = True,
            with_stack = True
        )as prof:
            start = time.perf_counter_ns() /1000000
            for i in range(nun_batches):
                input = inputs[i].to(device)
                with torch.no_grad():
                    output = model(input)
                    torch.cuda.synchronize()
                    output = output.cpu()
                prof.step()   
            print(opt.model_version, " total eval time: ", time.perf_counter_ns() /1000000 -start)
            #print(prof.key_averages().table(sort_by="cuda_time_total"))
    else:
        batch_times = []
        start = time.perf_counter_ns()
        for i in range(nun_batches):
            input = inputs[i].to(device)
            with torch.no_grad():
                output = model(input)
                torch.cuda.synchronize()
                output = output.cpu()  
                # Calcula el tiempo transcurrido para el batch actual y lo almacena
            batch_time = (time.perf_counter_ns() - start) / 1_000_000  # Convertir a milisegundos
            batch_times.append(batch_time)

        # Cálculo del tiempo promedio y máximo
        avg_time = sum(batch_times) / len(batch_times)
        max_time = max(batch_times)
        # Muestra el tiempo total, promedio y máximo
        print(f"{opt.model_version} total eval time: {sum(batch_times):.2f} ms, avg: {avg_time:.2f} ms, max: {max_time:.2f} ms")
        
    return

def validate(opt, val_loader, model, criterion, print_freq, batch_size):
    batch_time_all = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # Supongamos que 'model' es tu modelo PyTorch
    size_MB = get_model_size_MB(opt)

    # Calculate 10% of total batches
    warmup_batches = int(0.1 * len(val_loader))
    
    # Initialize the maximum and minimum processing time after warm-up
    max_time_all = 0
    min_time_all = float('inf')

    num_batches_to_process = int(1 * len(val_loader))

    if opt.profile:
        tracing_schedule = schedule(skip_first=0, wait=0, warmup=2, active=10, repeat=1)
        #trace_handler = tensorboard_trace_handler(dir_name=opt.log_dir)
        trace_handler = tensorboard_trace_handler(dir_name=os.path.abspath(opt.log_dir))
        with profile(
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule = tracing_schedule,
            on_trace_ready = trace_handler,
            profile_memory = True,
            record_shapes = True,
            with_stack = True
        )as prof:
            for i, (input, target) in enumerate(val_loader):

                if i >= num_batches_to_process:
                    break
                # Comprobar el tamaño del lote
                if input.size(0) != batch_size:
                    if not opt.less:
                        print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
                    break
                target = target.to(device)

                start_all = time.time() # start time, moving data to gpu
                input = input.to(device)
                with torch.no_grad():
                    output = model(input)
                    output_cpu = output.cpu() # con proposito de calcular el tiempo que tarda en volver a pasar la data a la cpu
                    all_time = (time.time() - start_all) * 1000  # Convert to milliseconds / time when the result pass to cpu again 
                    loss = criterion(output, target)
                prof.step()   
                #print(prof.key_averages().table(sort_by="cuda_time_total"))

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1[0], input.size(0))
                top5.update(prec5[0], input.size(0))

                # measure elapsed time in milliseconds and ignore first 10% batches
                if i >= warmup_batches:
                    
                    batch_time_all.update(all_time)
                    # Update the maximum and minimum processing time if necessary
                    max_time_all = max(max_time_all, all_time)
                    min_time_all = min(min_time_all, all_time)

                if not opt.less:
                    if i % print_freq == 0:
                        print('Test: [{0}/{1}]\t'
                                'Time {batch_time_all.val:.1f} ms ({batch_time_all.avg:.1f} ms)\t'
                                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                            i, len(val_loader), batch_time_all=batch_time_all, loss=losses,
                            top1=top1, top5=top5))
    else:
        for i, (input, target) in enumerate(val_loader):
            if i >= num_batches_to_process:
                break
            # Comprobar el tamaño del lote
            if input.size(0) != batch_size:
                if not opt.less:
                    print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
                break
            target = target.to(device)

            start_all = time.time() # start time, moving data to gpu
            input = input.to(device)
            with torch.no_grad():
                output = model(input)
                output_cpu = output.cpu() # con proposito de calcular el tiempo que tarda en volver a pasar la data a la cpu
                all_time = (time.time() - start_all) * 1000  # Convert to milliseconds / time when the result pass to cpu again 
                loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time in milliseconds and ignore first 10% batches
            if i >= warmup_batches:
                
                batch_time_all.update(all_time)
                # Update the maximum and minimum processing time if necessary
                max_time_all = max(max_time_all, all_time)
                min_time_all = min(min_time_all, all_time)

            if not opt.less:
                if i % print_freq == 0:
                    print('Test: [{0}/{1}]\t'
                            'Time {batch_time_all.val:.1f} ms ({batch_time_all.avg:.1f} ms)\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time_all=batch_time_all, loss=losses,
                        top1=top1, top5=top5))
    
    #----------------------------------------------------------------------------------------------------------#
    #                           INTERVALO DE CONFIANZA 95%                                                     #
    #----------------------------------------------------------------------------------------------------------#
    # Supongamos que tienes tus datos de tiempos en una lista llamada data
    data = batch_time_all.values  # Esta es la lista de tus tiempos
    # Estimación de parámetros de la distribución gamma
    alpha_hat, loc_hat, beta_hat = stats.gamma.fit(data, floc=0)  # Forzamos a que la localización (loc) sea 0
    mean_gamma = alpha_hat * beta_hat
    # Calcular el intervalo de confianza del 95%
    ci_lower = stats.gamma.ppf(0.025, alpha_hat, scale=beta_hat)
    ci_upper = stats.gamma.ppf(0.975, alpha_hat, scale=beta_hat)

    # Margen de error inferior y superior
    margin_error_lower = mean_gamma - ci_lower
    margin_error_upper = ci_upper - mean_gamma

    infxs = (opt.batch_size*1000 ) / batch_time_all.avg # inferenicas por segundo
    infxs_me_up = (opt.batch_size*1000 ) / batch_time_all.avg - (opt.batch_size*1000 ) / (batch_time_all.avg + margin_error_upper) # marginal error inferencias por segundo intervalo de confianza 95%
    infxs_me_low = (opt.batch_size*1000 ) / (batch_time_all.avg - margin_error_lower) -  (opt.batch_size*1000 ) / batch_time_all.avg # marginal error inferencias por segundo intervalo de confianza 95%
    
    #----------------------------------------------------------------------------------------------------------#
    #                                                                                                          #
    #----------------------------------------------------------------------------------------------------------#

    total_parametros = get_parametros(opt)
    total_capas = get_layers(opt)
    if not opt.non_verbose:
        print("|  Model          | inf/s +-95% | Latency (ms) +-95%|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|")
        print("|-----------------|-------------|-----------------------|-----------|----------------------|---------------------|---------|------------|")
    print("| {:<15} |  {:}  +{:} -{:} | {:>5.1f} / {:<6.1f}  +{:.1f} -{:.1f} |  {:<9.1f} | {:<20.2f} | {:<19.2f} | {:<6}  | {:<9}  |".format(
        opt.model_version, 
        number_formater(infxs) ,number_formater(infxs_me_up) ,number_formater(infxs_me_low),
        batch_time_all.avg, max_time_all, margin_error_upper,margin_error_lower,
        size_MB, 
        top1.avg, top5.avg,
        total_capas,total_parametros))

    if opt.histograma:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.hist(batch_time_all.values, bins=50, color='blue', alpha=0.7)
        plt.title('Distribución de batch_time_all')
        plt.xlabel('Tiempo (ms)')
        plt.ylabel('Frecuencia')
        plt.grid(True)
        plt.savefig("diosmelibre.pdf", bbox_inches='tight',format='pdf')

    return top1.avg, top5.avg

def number_formater(numero):
    return "{:,.1f}".format(numero).replace(",", "X").replace(".", ",").replace("X", ".")

def get_model_size_MB(opt):
    if opt.trt:
        return os.path.getsize(opt.engine) / (1024 * 1024) 
    else:
        if opt.network == 'yolo':
            return os.path.getsize(opt.weights) / (1024 * 1024) 
        import glob
        hub_dir = torch.hub.get_dir()
        # Buscar archivos que coincidan con el patrón
        matching_files = glob.glob(str(hub_dir) + '/checkpoints/' + opt.network + '*.pth')
        
        # Si hay al menos un archivo que coincide, devolver la ruta del primero
        if matching_files:
            model_path = matching_files[0]

            return os.path.getsize(model_path) / (1024 * 1024) 
        else:
            return None 

def get_parameters_vanilla(opt, model):
    total_capas = sum(1 for _ in model.modules())
    total_parametros = sum(p.numel() for p in model.parameters())
    #summary(model, (3,224,224)) ## summary modelo pth o pt segun pytorch
    return total_capas, total_parametros

def get_layers(opt):
    # para que funcione como sudo es necesario correr desde el path del enviroment env/bin/polygraphy
    if opt.trt:
        cmd = f"env/bin/polygraphy inspect model {opt.engine}"
    else:
        cmd = f"env/bin/polygraphy inspect model {(opt.engine).replace('.engine', '.onnx')} --display-as=trt"

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
        cmd = f"env/bin/python post_processing/param_counter.py --engine ../{opt.engine}"
    else:
        cmd = f"env/bin/onnx_opcounter {(opt.engine).replace('.engine', '.onnx')}"

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
    parser.add_argument('--profile', action='store_true',help='profiles the validation run with torch profiler')
    parser.add_argument('--less', action='store_true',help='print less information')
    parser.add_argument('--non_verbose', action='store_true',help='no table header and no gpu information')
    parser.add_argument('--model_version', default='Vanilla',help='model name in the table output (validation): Vanilla, TRT fp32, TRT fp16 TRT int8')
    parser.add_argument('--histograma', action='store_true',help='guarda una figura con el histograma de los tiempos x inferencia de cada batch')
    parser.add_argument('--log_dir', default='log/log_vnll', help='path to log dir for pytorch profiler')
   
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)