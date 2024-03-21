import torch.nn as nn
import torch.nn.init as init

class JuanjoOld(nn.Module):
    def __init__(self, nx, M, nu, L, leaky):
        super(JuanjoOld, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(nx, M))
        self.layers.append(nn.LeakyReLU(negative_slope=leaky,inplace=True))
        
        for _ in range(L-1):
            self.layers.append(nn.Linear(M, M))
            self.layers.append(nn.LeakyReLU(negative_slope=leaky,inplace=True))
        
        self.layers.append(nn.Linear(M, nu))
        self.layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

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

class JuanjoNet(nn.Module):
    def __init__(self, nx, M, nu, L):
        super(JuanjoNet, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(nx, M))
        self.layers.append(nn.ReLU(inplace=True))
        
        for _ in range(L-1):
            self.layers.append(nn.Linear(M, M))
            self.layers.append(nn.ReLU(inplace=True))
        
        self.layers.append(nn.Linear(M, nu))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def inicializar_pesos(self):
        for layer in self.layers:
            if hasattr(layer, 'weight'):
                init.xavier_uniform_(layer.weight)