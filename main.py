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
        engine_path = os.path.join(current_directory,opt.weights)
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
        validate(val_loader, model, criterion, opt.print_freq,opt.batch_size)

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
        evaluate(model)
    return

#-------------------------------------------
# Compare the outputs of the models given a 
# random input as the paper says
#-------------------------------------------
def compare(model, Engine, batch_size, rtol):
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    
    cumulative_absolute_error = 0.0
    closeness_count = 0

    num_batches = 1000
    for i in range(num_batches):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)

        input = input.to(device)
        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_trt = Engine(input)

        # Calculate the absolute error for the entire output vectors
        absolute_error = torch.abs(output_vanilla - output_trt).sum().item()
        cumulative_absolute_error += absolute_error

        # Calculate closeness for the entire output vectors
        close_values = torch.isclose(output_vanilla, output_trt, rtol=rtol).sum().item()
        closeness_count += close_values

    # Get average absolute error
    avg_absolute_error = cumulative_absolute_error / (num_batches * output_vanilla.size(1))

    # Convert closeness count to percentage
    total_values = num_batches * output_vanilla.size(1)
    closeness_percentage = (closeness_count / total_values) * 100

    print("Error Absoluto Promedio para todo el vector de resultados:", avg_absolute_error)
    print(f"Porcentaje de valores cercanos (usando torch.isclose) para todo el vector de resultados: {closeness_percentage:.4f}%")

#-------------------------------------------
# Compare the outputs of the models given a 
# random inputs showing the top5 MSE errors
#-------------------------------------------
def compare2(model, Engine, batch_size, rtol):
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    top_n = 5
    disagreements = np.zeros(top_n,dtype=np.int32)  # Track the number of disagreements for top1 to top5
    mse_errors = np.zeros(top_n,dtype=np.float64)

    num_batches = 1000

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

            #print("socre_v: ", score_v, " score_t: ", score_t)

            # Accumulate squared differences
            mse_errors[j] += (score_v - score_t) ** 2

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

def evaluate(model):
    model.eval()    
    return

def validate(val_loader, model, criterion, print_freq, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    # Calculate 10% of total batches
    warmup_batches = int(0.1 * len(val_loader))
    
    # Initialize the maximum and minimum processing time after warm-up
    max_time_post_warmup = 0
    min_time_post_warmup = float('inf')

    for i, (input, target) in enumerate(val_loader):
        # Comprobar el tamaño del lote
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        target = target.to(device)
        input = input.to(device)
        
        with torch.no_grad():
            # compute output
            end = time.time()
            output = model(input)
            elapsed_time = (time.time() - end) * 1000  # Convert to milliseconds
            loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top5.update(prec5[0], input.size(0))

        # measure elapsed time in milliseconds and ignore first 10% batches
        if i >= warmup_batches:
            batch_time.update(elapsed_time)
            # Update the maximum and minimum processing time if necessary
            max_time_post_warmup = max(max_time_post_warmup, elapsed_time)
            min_time_post_warmup = min(min_time_post_warmup, elapsed_time)

        if i % print_freq == 0:
            print('Test: [{0}/{1}]\t'
                    'Time {batch_time.val:.1f} ms ({batch_time.avg:.1f} ms)\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                    'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                i, len(val_loader), batch_time=batch_time, loss=losses,
                top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(f' * Minimum Time After Warm-up {min_time_post_warmup:.1f} ms')
    print(' * Average Time Per Batch {batch_time.avg:.1f} ms'.format(batch_time=batch_time))
    print(f' * Maximum Time After Warm-up {max_time_post_warmup:.1f} ms')

    return top1.avg, top5.avg

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='val_images/', help='path to dataset')
    parser.add_argument('--batch_size', default = 1, type=int,help='batch size to train')
    parser.add_argument('--weights', default = 'weights/best.engine', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
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

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)