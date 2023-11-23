import torch
import torch.nn as nn
import time
from torchinfo import summary

class ReLUNet(nn.Module):
    def __init__(self, num_layers=18, num_classes=1000):
        super(ReLUNet, self).__init__()
        
        # Inicializar módulos de la red
        layers = []

        for _ in range(num_layers):
            layers.append(nn.ReLU())

        self.relu_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu_layers(x)
        return x
    
def evaluate(model,batch_size=1):
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('CUDA is not available.')
    else:
        print('CUDA is available.')

    global device
    device = torch.device("cuda:0" if train_on_gpu else "cpu")

    model.to(device)
    #summary(model, (3,224,224))

    num_batches = 10000
    total_time = 0
    for i in range(num_batches):
        start_time = time.time()
        
        torch.manual_seed(i)
        input = torch.rand(batch_size, 3, 224, 224)
        input = input.to(device)
        
        with torch.no_grad():
            output = model(input)
            torch.cuda.synchronize()
            output = output.cpu()
        
        end_time = time.time()
        total_time += (end_time - start_time)

    average_time = total_time / num_batches
    print(f"Tiempo promedio por ciclo: {average_time} segundos")
    return

# Crear una instancia de la red
net = ReLUNet()
evaluate(net)


from engine import TRTModule 
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
engine_path = os.path.join(current_directory,"best.engine")

Engine = TRTModule(engine_path, device)
Engine.set_desired(['outputs'])
evaluate(Engine)
