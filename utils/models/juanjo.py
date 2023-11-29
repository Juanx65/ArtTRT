import torch.nn as nn

class CustomQuantizedNet(nn.Module):
    def __init__(self, nx, M, nu, L, leaky):
        super(CustomQuantizedNet, self).__init__()
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

        
