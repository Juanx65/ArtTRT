import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data

from utils.models.juanjo import CustomQuantizedNet
from utils.models.resnet import resnet18, resnet50, resnet101, resnet152
from utils.data_loader import data_loader
from utils.helper import AverageMeter, accuracy, adjust_learning_rate

from torchinfo import summary

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

#torch.backends.cudnn.enabled = False

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU')
else:
    print('CUDA is available. Training on GPU')

device = torch.device("cuda:0" if train_on_gpu else "cpu")
#device = torch.device("cpu")



# para entrenar:
# python main_own_trained.py --lr 0.001 --wd 1e-6 --epoch 100 --batch_size 32

#para evaluar (imprimir curva r2):
#python main_own_trained.py --batch_size 1 -e

#para validar (comprobar MSE y loss):
#python main_own_trained.py --batch_size 1 -v

# aplciar -trt para usar el engine a los comandos de evaluar y validar

def main_train_eval(opt):
    #evaluate()
    #return
    global device
    
    """ 
    if(opt.resnet50):
        model = resnet50()
    else:
        model = resnet18() 
    """
    # Parámetros
    nx = 2
    M = 258
    nu = 1
    L = 1
    leaky = 0.00390625

    # Crear la red
    model = CustomQuantizedNet(nx, M, nu, L, leaky)
    summary(model)
    #print("parameters: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.to(device)

    # define loss and optimizer
    criterion = nn.MSELoss().to(device)#nn.CrossEntropyLoss().to(device)
    
    # obs: l2_regularization = opt.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)#wd == l2
    global l1_lambda
    l1_lambda = opt.weight_decay

    """     
    optimizer = optim.SGD(model.parameters(), lr=opt.lr,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    """

    # Data loading
    #train_loader, val_loader = data_loader(opt.dataset, opt.batch_size, opt.workers, opt.pin_memmory)

    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from torch.utils.data import DataLoader, TensorDataset

    # Carga de datos
    data = pd.read_csv('datasets/empc_data_ref_mixed.csv')
    #data = data.sample(frac=0.8)
    data.describe()
    
    #print(data.describe())

    #normalizacion
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    X = torch.tensor(data[:,:2],dtype=torch.float)
    y = torch.tensor(data[:,2], dtype=torch.float)

    # Divide los datos: 70% entrenamiento, 30% validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
    # DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    ## cargar el modelo

    if opt.trt:
        from utils.engine import TRTModule #if not done here, unable to train
        current_directory = os.path.dirname(os.path.abspath(__file__))
        engine_path = os.path.join(current_directory,opt.engine)
        Engine = TRTModule(engine_path,device)
        Engine.set_desired(['outputs'])
        model = Engine
    else:    
        model = torch.load(opt.weights)
    model.to(device) 

    ## validar, evaluar o entrenar
    if opt.validate:
        validate(val_loader, model, criterion, opt.print_freq,opt.batch_size)
        return
    
    if opt.evaluate:
        evaluate(val_loader, model, opt.batch_size)
        return

    # entrenar
    for epoch in range(0, opt.epochs):
        #adjust_learning_rate(optimizer, epoch, opt.lr)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, opt.print_freq,opt.batch_size)
   
        # evaluate on validation set val_loader
        loss = validate(val_loader, model, criterion, opt.print_freq,opt.batch_size)
        # remember the best loss and save checkpoint
        
        from math import inf
        best_loss = inf
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if is_best:
            torch.save(model, opt.weights) 
       

def l1_penalty(parameters):
    return sum(p.abs().sum() for p in parameters)

def train(train_loader, model, criterion, optimizer, epoch, print_freq, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    mses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break
        
        target = target.unsqueeze(1)
        input, target = input.to(device), target.to(device)

        # compute output
        output = model(input)

        loss = criterion(output, target)
        mses.update(loss.item(),input.size(0))
        #print("output / target / loss: ", output.item()," / ", target[0].item(), " / ", loss.item())
        l1_loss = l1_penalty(model.parameters())
        loss = loss + l1_lambda * l1_loss
        # measure accuracy and record loss
        prec1, _ = accuracy(output.data, target, topk=(1, 1))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        #top5.update(prec1[0], input.size(0))

        # compute gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                  'MSE {mse.val:.6f} ({mse.avg:.6f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                epoch, i, len(train_loader), batch_time=batch_time, loss=losses,mse=mses, top1=top1))


def validate(val_loader, model, criterion, print_freq, batch_size):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        # Comprobar el tamaño del lote
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break
        target = target.unsqueeze(1)
        target = target.to(device)#cuda(async=True)
        input = input.to(device)#cuda(async=True)
        with torch.no_grad():
            # compute output
            output = model(input)

            loss = criterion(output, target)
            l1_loss = l1_penalty(model.parameters())
            loss = loss + l1_lambda * l1_loss

            # measure accuracy and record loss
            prec1, _ = accuracy(output.data, target, topk=(1, 1))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.6f} ({batch_time.avg:.6f})\t'
                      'Loss {loss.val:.6f} ({loss.avg:.6f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1))

    print(' * Prec@1 {top1.avg:.3f} '.format(top1=top1))
    print(' * Average Time Per Batch {batch_time.avg:.3f}'.format(batch_time=batch_time))
    return losses.avg

from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def evaluate(val_loader, model, batch_size):
    time_batch = AverageMeter()
    all_outputs = []
    all_targets = []
    times = []  # Lista para almacenar tiempos
    warmup_batches = int(0.1 * len(val_loader))

    #starter = torch.cuda.Event(enable_timing=True)
    #ender = torch.cuda.Event(enable_timing=True)

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if input.size(0) != batch_size:
            print(f"Deteniendo la evaluación. Tamaño del lote ({input.size(0)}) no es igual a batch_size ({batch_size}).")
            break

        target = target.unsqueeze(1)
        start_time = time.time()  # Iniciar el cronómetro

        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            #starter.record()
            output = model(input)
            #ender.record()
        
        #model_time = starter.elapsed_time(ender)

        #torch.cuda.synchronize()
        end_time = time.time()  # Detener el cronómetro

        if i >= warmup_batches:  # Guardar datos después del período de calentamiento
            times.append((end_time - start_time)*1000)  # Guardar el tiempo transcurrido
            time_batch.update((end_time - start_time)*1000)
            all_outputs.extend(output.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Calcular R^2 y graficar
    fig, axs = plt.subplots()

    # Graficar ajuste de valores
    axs.scatter(all_targets, all_outputs, color='blue', s=2)
    axs.set(xlabel="Valor real", ylabel="Valor DNN", title="Ajuste de valor\n R2 = {:.2f}".format(r2_score(all_targets, all_outputs)))
    axs.plot([0, 1],[0, 1], color='red')
    axs.plot([0, 1], [0.5, 0.5], color='grey', linestyle='dashed')
    axs.grid()
    fig.savefig("line_plot_r2_juanjo.png")
    
    print("timepo avg: ", time_batch.avg)
    fig, axs = plt.subplots()
    # Graficar tiempos de cálculo
    axs.plot(times, color='blue')
    axs.plot([time_batch.avg for i in times], color='green')
    axs.set(xlabel="Número de muestra", ylabel="Tiempo (ms)", title="Tiempo de cálculo por muestra")
    axs.grid()
    fig.savefig("times.png")

    plt.tight_layout()
    plt.show()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='dataset/', help='path to dataset')
    parser.add_argument('--batch_size', default = 32, type=int,help='batch size to train')
    parser.add_argument('--epochs', default = 90, type=int,help='epoch to train')
    parser.add_argument('--weight-decay', '--wd', default=1e-6, type=float, help='Weight decay (default: 1e-4)')
    parser.add_argument('--momentum', default = 0.9, type=float,help='momentum')
    parser.add_argument('--lr', default = 0.001, type=float, help='learning rate')
    parser.add_argument('--weights', default = 'weights/best.pth', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('--engine', default = 'weights/best.engine', type=str, help='directorio y nombre de archivo de donse se guardara el mejor peso entrenado')
    parser.add_argument('-m','--pin_memmory', action='store_true',help='use pin memmory')
    parser.add_argument('-j', '--workers', default=4, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',help='evaluate model on validation set')
    parser.add_argument('-v', '--validate', dest='validate', action='store_true',help='validate model on validation set')
    parser.add_argument('--print-freq', '-f', default=10, type=int, metavar='N',help='print frequency (default: 10)')
    parser.add_argument('-trt','--trt', action='store_true',help='evaluate model on validation set al optimizar con tensorrt')
    parser.add_argument('-rn50','--resnet50', action='store_true',help='use ResNet50 as model')
    parser.add_argument('-rn18','--resnet18', action='store_true',help='use ResNet18 as model')

    opt = parser.parse_args()
    return opt

def main(opt):
    main_train_eval(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)