import argparse
import os
import time

import torch
import torch.nn as nn
import torch.utils.data

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
        model = torch.hub.load('pytorch/vision:v0.15.2', opt.network, weights=f'ResNet{opt.network[6:]}_Weights.DEFAULT')

    model.to(device)

    if opt.validate:
        val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory)
        criterion = nn.CrossEntropyLoss().to(device)
        validate(val_loader, model, criterion, opt.print_freq,opt.batch_size)

    elif opt.compare:
        if opt.val_dataset:
            val_loader = val_data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory)
            compare_val(val_loader, model, Engine, opt.batch_size, opt.rtol)
        else:
            compare(model,Engine,opt.batch_size, opt.rtol)

    else:
        evaluate(model, opt.batch_size)
    return

def compare(model, Engine, batch_size, rtol):
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    
    for i in range(1):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)

        input = input.to(device)
        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_trt = Engine(input)

        # pasamos a cpu y Convierte los tensores a arrays de NumPy
        output_vanilla = output_vanilla.cpu().numpy()
        output_trt = output_trt.cpu().numpy()

        #print("output vanilla: ", output_vanilla)
        #print("output trt: ", output_trt)

        print("output vanilla shape: ", output_vanilla.shape)
        print("output trt shape: ", output_trt.shape)
        rtol = rtol
        atol= 1e-8
        contador_falses = 0  # Inicializamos el contador
        print("absolute(a - b) <= (atol + rtol * absolute(b))")
        for j in range(len(output_vanilla[0])):
            diferencia = np.abs(output_vanilla[0,j] - output_trt[0,j])
            umbral = atol + rtol * np.abs(output_trt[0,j])

            print("resta elemento", j, ":", diferencia, " <= ", umbral, ":", diferencia <= umbral)
            
            # Si la condición no se cumple (es decir, es False), incrementamos el contador
            if not (diferencia <= umbral):
                contador_falses += 1

        print("Elementos no iguales:", contador_falses)

        # Usa la función numpy.isclose() para comparar los arrays
        close_elements = np.isclose(output_vanilla, output_trt, rtol=rtol, atol=atol)

        # Reporta el porcentaje de elementos no iguales
        non_equal_elements = np.size(close_elements) - np.sum(close_elements)
        percentage_non_equal = (non_equal_elements / np.size(close_elements)) * 100

        print(f"Porcentaje de elementos no iguales: {percentage_non_equal:.2f}%")
    return

def compare_val(val_loader, model, Engine, batch_size, rtol):
    # switch to evaluate mode
    model.eval()
    Engine.eval()
    
    for i, (input, target) in enumerate(val_loader):

        input = input.to(device)
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break
        with torch.no_grad():
            # compute output
            output_vanilla = model(input)
            output_trt = Engine(input)

        # pasamos a cpu y Convierte los tensores a arrays de NumPy
        output_vanilla = output_vanilla.cpu().numpy()
        output_trt = output_trt.cpu().numpy()

        #print("output vanilla: ", output_vanilla)
        #print("output trt: ", output_trt)

        print("output vanilla shape: ", output_vanilla.shape)
        print("output trt shape: ", output_trt.shape)
        rtol = rtol
        atol= 1e-8
        contador_falses = 0  # Inicializamos el contador
        print("absolute(a - b) <= (atol + rtol * absolute(b))")
        for j in range(len(output_vanilla[0])):
            diferencia = np.abs(output_vanilla[0,j] - output_trt[0,j])
            umbral = atol + rtol * np.abs(output_trt[0,j])

            #print("resta elemento", j, ":", diferencia, " <= ", umbral, ":", diferencia <= umbral)
            
            # Si la condición no se cumple (es decir, es False), incrementamos el contador
            if not (diferencia <= umbral):
                contador_falses += 1

        print("Elementos no iguales:", contador_falses)

        # Usa la función numpy.isclose() para comparar los arrays
        close_elements = np.isclose(output_vanilla, output_trt, rtol=rtol, atol=atol)

        # Reporta el porcentaje de elementos no iguales
        non_equal_elements = np.size(close_elements) - np.sum(close_elements)
        percentage_non_equal = (non_equal_elements / np.size(close_elements)) * 100

        print(f"Porcentaje de elementos no iguales: {percentage_non_equal:.2f}%")
        break
    return



def evaluate(model, batch_size):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()
    
    #previous_input = None  # Inicializamos esta variable para guardar el input anterior

    end = time.time()
    for i in range(100):
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224) # generamos un input random [0,1)

        """ 
        # Valores de media y desviación estándar
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        # Normalizar
        input = (input - mean) / std
        #print("input :", input)
        #break

        # Si previous_input no es None (es decir, no es la primera iteración), calculamos la diferencia
        if previous_input is not None:
            difference = input - previous_input
            l2_norm = torch.norm(difference).item()
            print("Diferencia L2 con el input anterior:", l2_norm) 
        
        # Actualizamos previous_input con el valor actual de input
        previous_input = input.clone()
        """


        input = input.to(device)
        with torch.no_grad():
            # compute output
            output = model(input)
            _, pred = output.topk(5, 1, largest=True, sorted=True) # ordenamos la prediccion top-1 (en imagnet hay 1000 clases)
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    print(' * Average Time Per Batch {batch_time.avg:.3f}'.format(batch_time=batch_time))
    return


def validate(val_loader, model, criterion, print_freq, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # Comprobar el tamaño del lote
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break
        #print("input :", input)
        #break
        target = target.to(device)#cuda(async=True)
        input = input.to(device)#cuda(async=True)
        with torch.no_grad():
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))

    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    print(' * Average Time Per Batch {batch_time.avg:.3f}'.format(batch_time=batch_time))

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
    parser.add_argument('-rtol','--rtol', default=1e-2,type=float, help='relative tolerance for the numpy.isclose() function')
    parser.add_argument('-vd','--val_dataset', action='store_true',help='compare the results of the vanilla model with the trt model using the validation dataset as inputs')

    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)