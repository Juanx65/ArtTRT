import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

import numpy as np

from utils.data_loader import val_data_loader
from utils.helper import AverageMeter, accuracy

from torch.profiler import profile, ProfilerActivity,schedule, tensorboard_trace_handler
#from torchsummary import summary

import subprocess
import re

import scipy.stats as stats

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

best_prec1 = 0.0

def main(opt):
    train_on_gpu = torch.cuda.is_available()
    if not opt.non_verbose:
        if not train_on_gpu:
            print('CUDA is not available.')
        else:
            print('CUDA is available.')

    global best_prec1, device
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    if opt.trt and not opt.compare_3:
        from utils.engine import TRTModule #if not done here, unable to train
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.engine)
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
            YOLOv8 = YOLO(opt.weights)
            model = YOLOv8.model.fuse()
        else:
            print("Red no reconocida.")
    model.to(device)

    if opt.validate:
        if "yolo" in  opt.network:
            val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory, do_normalize=False)
        else:
            val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory)
        criterion = nn.CrossEntropyLoss().to(device)
        if opt.profile:
            validate_profile(val_loader, model)
        else:
            validate(opt, val_loader, model, criterion, opt.print_freq,opt.batch_size)

    elif opt.compare:
        if opt.val_dataset:
            if "yolo" in  opt.network:
                val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory, do_normalize=False)
            else:
                val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory)
            compare_val(val_loader, model, Engine, opt.batch_size, opt.rtol)
        elif opt.compare_3:
            from utils.engine import TRTModule
            current_directory = os.path.dirname(os.path.abspath(__file__))
            engine_path_1 = os.path.join(current_directory,"weights/best_fp32.engine")
            engine_path_2 = os.path.join(current_directory,"weights/best_fp16.engine")
            engine_path_3 = os.path.join(current_directory,"weights/best_int8.engine")
            Engine_fp32 = TRTModule(engine_path_1,device)
            Engine_fp16 = TRTModule(engine_path_2,device)
            Engine_int8 = TRTModule(engine_path_3,device)
            Engine_fp32.set_desired(['outputs'])
            Engine_fp16.set_desired(['outputs'])
            Engine_int8.set_desired(['outputs'])
            compare_models(model,Engine_fp32,Engine_fp16,Engine_int8,opt.batch_size, opt.rtol)
        else:
            compare(model,Engine,opt.batch_size, opt.rtol)

    else:
        print("nsight evaluation test.")
        evaluate(opt, model)
    return

def compare_models(model, Engine_fp32,Engine_fp16,Engine_int8, batch_size,rtol):
    model.eval()
    Engine_fp32.eval()
    Engine_fp16.eval()
    Engine_int8.eval()

    top_n = 10
    disagreements_fp32 = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    disagreements_fp16 = np.zeros(top_n,dtype=np.int32) 
    disagreements_int8 = np.zeros(top_n,dtype=np.int32) 

    mae_errors_fp32 = np.zeros(top_n,dtype=np.float64)
    mae_errors_fp16 = np.zeros(top_n,dtype=np.float64)
    mae_errors_int8 = np.zeros(top_n,dtype=np.float64)

    num_batches = 5000

    closeness_count_fp32 = 0
    closeness_count_fp16 = 0
    closeness_count_int8 = 0

    for i in range(num_batches):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)

        input = input.to(device)
        
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_fp32 = Engine_fp32(input)
            output_fp16 = Engine_fp16(input)
            output_int8 = Engine_int8(input)

        close_values_fp32 = torch.isclose(output_vanilla, output_fp32, rtol=rtol).sum().item()
        close_values_fp16 = torch.isclose(output_vanilla, output_fp16, rtol=rtol).sum().item()
        close_values_int8 = torch.isclose(output_vanilla, output_int8, rtol=rtol).sum().item()

        closeness_count_fp32 += close_values_fp32
        closeness_count_fp16 += close_values_fp16
        closeness_count_int8 += close_values_int8
            
         # Get top n classes and their scores for each model
        top_scores_vanilla, top_indices_vanilla = torch.topk(output_vanilla, top_n)
        top_scores_fp32, top_indices_fp32 = torch.topk(output_fp32, top_n)
        top_scores_fp16, top_indices_fp16 = torch.topk(output_fp16, top_n)
        top_scores_int8, top_indices_int8 = torch.topk(output_int8, top_n)

        # Compare the top classes and their scores for each position from top1 to top5
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_fp32[0])):
            if idx_v.item() != idx_t.item():
                disagreements_fp32[j] += 1
            # Get scores of the top class of Engine (trt) in both models
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_fp32[0, idx_t].item()
            mae_errors_fp32[j] += np.abs(score_v - score_t)
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_fp16[0])):
            if idx_v.item() != idx_t.item():
                disagreements_fp16[j] += 1
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_fp16[0, idx_t].item()
            mae_errors_fp16[j] += np.abs(score_v - score_t)
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_int8[0])):
            if idx_v.item() != idx_t.item():
                disagreements_int8[j] += 1
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_int8[0, idx_t].item()
            mae_errors_int8[j] += np.abs(score_v - score_t)


    # Get average MSE for each position
    mae_errors_fp32 = [error / num_batches for error in mae_errors_fp32]
    mae_errors_fp16 = [error / num_batches for error in mae_errors_fp16]
    mae_errors_int8 = [error / num_batches for error in mae_errors_int8]

    # Convert disagreements to percentages
    total_images = num_batches * batch_size
    disagreement_percentages_fp32 = [(disagreement / total_images) * 100 for disagreement in disagreements_fp32]
    disagreement_percentages_fp16 = [(disagreement / total_images) * 100 for disagreement in disagreements_fp16]
    disagreement_percentages_int8 = [(disagreement / total_images) * 100 for disagreement in disagreements_int8]
    # Print header
    print("|  Rank  | MAE / Disg  fp32 | MAE / Disg  fp16 | MAE / Disg  int8 |")
    print("|--------|------------------|------------------|------------------|")

    # Print each row for ranks 1 to 10
    for rank in range(top_n):
        print("| {:6} | {:<16} | {:<16} | {:<16} |".format(
            rank + 1, 
            "{:.4f} / {:.2f}".format(mae_errors_fp32[rank], disagreement_percentages_fp32[rank]),
            "{:.4f} / {:.2f}".format(mae_errors_fp16[rank], disagreement_percentages_fp16[rank]),
            "{:.4f} / {:.2f}".format(mae_errors_int8[rank], disagreement_percentages_int8[rank])
        ))
    
    total_values = num_batches * output_vanilla.size(1)
    closeness_percentage_fp32 = (closeness_count_fp32 / total_values) * 100
    closeness_percentage_fp16 = (closeness_count_fp16 / total_values) * 100
    closeness_percentage_int8 = (closeness_count_int8 / total_values) * 100

    # Print header
    print("| Vanilla VS | equality (%) |")
    print("|------------|--------------|")
    print("| TRT fp32 | {:.2f} |".format(closeness_percentage_fp32))
    print("| TRT fp16 | {:.2f} |".format(closeness_percentage_fp16))
    print("| TRT int8 | {:.2f} |".format(closeness_percentage_int8))

    return

def compare(model, Engine, batch_size, rtol):
    #from tabulate import tabulate
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    top_n = 10
    disagreements = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    mae_errors = np.zeros(top_n,dtype=np.float64)

    num_batches = 5000

    closeness_count = 0
    for i in range(num_batches):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)

        input = input.to(device)
        
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_trt = Engine(input)

        close_values = torch.isclose(output_vanilla, output_trt, rtol=rtol).sum().item()
        closeness_count += close_values
            
         # Get top n classes and their scores for each model
        top_scores_vanilla, top_indices_vanilla = torch.topk(output_vanilla, top_n)
        top_scores_trt, top_indices_trt = torch.topk(output_trt, top_n)

        # Compare the top classes and their scores for each position from top1 to top5
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_trt[0])):
            if idx_v.item() != idx_t.item():
                disagreements[j] += 1
            
            # Get scores of the top class of Engine (trt) in both models
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_trt[0, idx_t].item()

            #print("socre_v: ", score_v, " score_t: ", score_t)

            # MEAN Absolute Error differences
            mae_errors[j] += np.abs(score_v - score_t)

        """ por si se necesita una tabla que muestre los top n scores de cada salida
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
        """

    # Get average MSE for each position
    mae_errors = [error / num_batches for error in mae_errors]

    # Convert disagreements to percentages
    total_images = num_batches * batch_size
    disagreement_percentages = [(disagreement / total_images) * 100 for disagreement in disagreements]

    print("Disagreements (in percentage) for each position:")
    for j, percentage in enumerate(disagreement_percentages, 1):
        print(f"Top {j}: {percentage:.4f}% disagreements")
    print("Mean Squared Error for each position:")
    for j, error in enumerate(mae_errors, 1):
        print(f"Top {j}: {error:.8f} MAE")

    total_values = num_batches * output_vanilla.size(1)
    closeness_percentage = (closeness_count / total_values) * 100
    print(f"Porcentaje de valores cercanos (usando torch.isclose) para todo el vector de resultados: {closeness_percentage:.4f}%")

    return

def compare_val(val_loader, model, Engine, batch_size, rtol=1e-3):
    # switch to evaluate mode
    model.eval()
    Engine.eval()

    top_n = 5
    disagreements = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    mse_errors = np.zeros(top_n,dtype=np.float64)

    for i, (input, target) in enumerate(val_loader):
        input = input.to(device)

        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        with torch.no_grad():
            # compute output
            output_vanilla = torch.sigmoid(model(input))
            output_trt = torch.sigmoid(Engine(input))

        # Get top 5 classes and their scores for each model
        _, top_indices_vanilla = torch.topk(output_vanilla, top_n)
        _, top_indices_trt = torch.topk(output_trt, top_n)

        # Compare the top classes and their scores for each position from top1 to top5
        for j, (idx_v, idx_t) in enumerate(zip(top_indices_vanilla[0], top_indices_trt[0])):
            if idx_v.item() != idx_t.item():
                disagreements[j] += 1
            
            # Get scores of the top class of Engine (trt) in both models
            score_v = output_vanilla[0, idx_t].item()
            score_t = output_trt[0, idx_t].item()

            # Accumulate squared differences
            mse_errors[j] += (score_v - score_t) ** 2

    num_batches = len(val_loader)
    # Get average MSE for each position
    mse_errors = [error / num_batches for error in mse_errors]

    # Convert disagreements to percentages
    total_images = num_batches * batch_size
    disagreement_percentages = [(disagreement / total_images) * 100 for disagreement in disagreements]

    print("Disagreements (in percentage) for each position:")
    for j, percentage in enumerate(disagreement_percentages, 1):
        print(f"Top {j}: {percentage:.4f}% disagreements")
    print("Mean Squared Error for each position:")
    for j, error in enumerate(mse_errors, 1):
        print(f"Top {j}: {error:.8f} MSE")

    return

def evaluate(opt, model):
    nun_batches = 10
    inputs= torch.rand(nun_batches,opt.batch_size, 3, 224, 224) # generamos un input random [0,1)

    tracing_schedule = schedule(skip_first=5, wait=5, warmup=2, active=2, repeat=1)
    trace_handler = tensorboard_trace_handler(dir_name=opt.log_dir, use_gzip=True)

    with profile(
        activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #schedule = tracing_schedule,
        on_trace_ready = trace_handler,
        profile_memory = True,
        record_shapes = True,
        with_stack = True
    )as prof:
        start = time.perf_counter_ns() /1000000
        for i in range(nun_batches):
            torch.manual_seed(i)
            #input = torch.rand(opt.batch_size, 3, 224, 224) # generamos un input random [0,1)
            input = inputs[i].to(device)
            with torch.no_grad():
                output = model(input)
                torch.cuda.synchronize()
                output = output.cpu()
            prof.step()   
        print(opt.model_version, " total eval time: ", time.perf_counter_ns() /1000000 -start)
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
    return

def validate(opt, val_loader, model, criterion, print_freq, batch_size):
    batch_time_all = AverageMeter()
    batch_time_model = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)

    # switch to evaluate mode
    model.eval()
    # Supongamos que 'model' es tu modelo PyTorch
    size_MB = get_model_size_MB(opt)

    # Calculate 10% of total batches
    warmup_batches = int(0.1 * len(val_loader))
    
    # Initialize the maximum and minimum processing time after warm-up
    max_time_all = 0
    min_time_all = float('inf')

    max_time_model = 0
    min_time_model = float('inf')

    num_batches_to_process = int(1 * len(val_loader))

    """
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        #record_shapes=True,
        #with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_vnll')) as prof:
    """
    pasa_marginal = 0
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
            starter.record()
            #start = time.time()
            output = model(input)
            #end = time.time()
            ender.record()

            torch.cuda.synchronize()

            model_time = starter.elapsed_time(ender)
            #model_time = (end - start) * 1000

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
            
            if( all_time > 2.94 or all_time < 1.83):
                pasa_marginal += 1  

            batch_time_all.update(all_time)
            batch_time_model.update(model_time)
            # Update the maximum and minimum processing time if necessary
            max_time_all = max(max_time_all, all_time)
            min_time_all = min(min_time_all, all_time)

            max_time_model = max(max_time_model, model_time)
            min_time_model = min(min_time_model, model_time)

        if not opt.less:
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                        'Time {batch_time_all.val:.1f} ms ({batch_time_all.avg:.1f} ms)\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time_all=batch_time_all, loss=losses,
                    top1=top1, top5=top5))
            
        #prof.step()   
        #print(prof.key_averages().table(sort_by="cuda_time_total"))
    
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
    
    #n = len(val_loader) - warmup_batches
    #print("n: ", n)
    #print("pasa marginal: ", pasa_marginal) # asegurarse de que sea menor del 5% para que el calculo sea correcto

    total_parametros = get_parametros(opt)
    total_capas = get_layers(opt)
    if not opt.non_verbose:
        print("|  Model          | inf/s +-95% | Latency-all (ms) +-95%|Latency-model (ms) |size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)| #layers | #parameters|")
        print("|-----------------|-------------|-----------------------|------------------------|-----------|----------------------|---------------------|---------|------------|")
    print("| {:<15} |  {:}  +{:} -{:} | {:>5.1f} / {:<6.1f}  +{:.1f} -{:.1f} | {:>6.1f} / {:<7.1f} |  {:<9.1f} | {:<20.2f} | {:<19.2f} | {:<6}  | {:<9}  |".format(
        opt.model_version, 
        number_formater(infxs) ,number_formater(infxs_me_up) ,number_formater(infxs_me_low),
        batch_time_all.avg, max_time_all, margin_error_upper,margin_error_lower,
        batch_time_model.avg, max_time_model,
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

def validate_profile(val_loader, model):
    
    num_batches_to_process = int(1/5 * len(val_loader))
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        #record_shapes=True,
        #with_stack=True,
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/log_trt')) as prof:

        #elapsed_time = 0
        for i, (input, target) in enumerate(val_loader):
            if i >= num_batches_to_process:
                break
            start= time.time()
            input = input.to(device)
            with torch.no_grad():
                output = model(input)
                output = output.cpu()
                #elapsed_time += (time.time() - start) * 1000 

            prof.step() 
    
    #print("avg time: ", elapsed_time/100, " ms")
    return

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
    parser.add_argument('-vd','--val_dataset', action='store_true',help='compare the results of the vanilla model with the trt model using the validation dataset as inputs')
    parser.add_argument('--profile', action='store_true',help='profiles the validation run with torch profiler')
    parser.add_argument('--compare_3', action='store_true',help='compare the results of the vanilla model with the trt model using random generated inputs')
    parser.add_argument('--less', action='store_true',help='print less information')
    parser.add_argument('--non_verbose', action='store_true',help='no table header and no gpu information')
    parser.add_argument('--model_version', default='Vanilla',help='model name in the table output (validation): Vanilla, TRT fp32, TRT fp16 TRT int8')
    parser.add_argument('--histograma', action='store_true',help='guarda una figura con el histograma de los tiempos x inferencia de cada batch')#'./log/log_vnll'}
    parser.add_argument('--log_dir', default='log/log_vnll', help='path to log dir for pytorch profiler')
   
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)