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

from torch.profiler import profile, record_function, ProfilerActivity

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.')
else:
    print('CUDA is available.')

device = torch.device("cuda:0" if train_on_gpu else "cpu")

best_prec1 = 0.0

def main(opt):
    global best_prec1, device

    if opt.trt:
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
        else:
            compare(model,Engine,opt.batch_size, opt.rtol)

    else:
        print("nsight evaluation test.")
        evaluate(model, opt.batch_size)
    return

def compare(model, Engine, batch_size, rtol):
    from tabulate import tabulate
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    top_n = 10
    disagreements = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    mae_errors = np.zeros(top_n,dtype=np.float64)

    num_batches = 10

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

def evaluate(model,batch_size):
    nun_batches = 10

    for i in range(nun_batches):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)
        input = input.to(device)
        with torch.no_grad():
            output = model(input)
            torch.cuda.synchronize()
            output = output.cpu()
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
    for i, (input, target) in enumerate(val_loader):

        if i >= num_batches_to_process:
            break
        # Comprobar el tamaño del lote
        if input.size(0) != batch_size:
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
            batch_time_all.update(all_time)
            batch_time_model.update(model_time)
            # Update the maximum and minimum processing time if necessary
            max_time_all = max(max_time_all, all_time)
            min_time_all = min(min_time_all, all_time)

            max_time_model = max(max_time_model, model_time)
            min_time_model = min(min_time_model, model_time)

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

    print("|  Model          |Latency-all (ms)|Latency-model (ms)|size (MB)  | accuracy (Prec@1) (%)|accuracy (Prec@5) (%)|")
    print("|-----------------|----------------|------------------|-----------|----------------------|---------------------|")
    print("| {:<15} | {:>4.1f} / {:<4.1f} | {:>4.1f} / {:<4.1f} | {:<7.1f} | {:<20.2f} | {:<19.2f} |".format(
        opt.network, 
        batch_time_all.avg, max_time_all, 
        batch_time_model.avg, max_time_model,
        size_MB, 
        top1.avg, top5.avg))

    return top1.avg, top5.avg

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

        
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='val_images/', help='path to dataset')
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

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)