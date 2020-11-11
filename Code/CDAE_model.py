import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torch.autograd import Variable
from torchsummary import summary



class CDAE(nn.Module):
    
    def __init__(self):
        super(CDAE, self).__init__()
        
        self.encoder = nn.Sequential(
            #input size: [batch_size, 1, H,W=[100,100]]
            nn.Conv2d(1, 10, kernel_size=15, stride=1, padding=7, bias=False),
            nn.Tanh(),
            # [batch_size, 10, [100,100]]
            nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)),
            # [batch_size, 10, [50, 50]]
            nn.Conv2d(10, 20, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
            # [batch_size, 20, [50, 50]]
            nn.MaxPool2d((5,5), (5,5)),
            # [batch_size, 20, [10,10]]
            nn.Conv2d(20, 30, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
            # [batch_size, 30, [10,10]]
            nn.Conv2d(30, 40, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
            # [batch_size, 40, [10,10]]
            
        )
        
        self.decoder = nn.Sequential(
            # [batch_size, 40, [10,10]]
            nn.ConvTranspose2d(40, 30, kernel_size=(5,5), stride=(5,5), bias=False),
            nn.Tanh(),
            # [batch_size, 30, [50, 50]]
            nn.ConvTranspose2d(30, 20, kernel_size=(2,2), stride=(2,2), bias=False),
            nn.Tanh(),
            # [batch_size, 20, [100,100]]
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh(),
            # [batch_size, 10, [100,100]]
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=1, padding=2, bias=False),
            nn.Tanh()
            # [batch_size, 1, [126,56]]
        )
        
    def forward(self, x):
        
        x = self.encoder(x)
        x = self.decoder(x)
        
        return x
