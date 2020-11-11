# %matplotlib inline
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('seaborn-white')

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchsummary import summary

from CDAE_model import *

import copy
import random
from datetime import datetime

class CDAE_trainer:
    def __init__(self,
                 model,
                 path,
                 filename,
                 criterion = nn.MSELoss(),  
                 optimizer=optim.Adam,
                 lr=0.001,
                 weight_decay=1e-5):
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.criterion = criterion
        self.optimizer_type = optimizer
        self.optimizer = self.optimizer_type(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.path = path
        self.filename=filename
        
    def train(self,
              mixed_input, # mixed sources spectrograms
              target,      # clean target source spectrograms
              n,           # datasets size (tot number of spectrograms)
              device,      # GPU vs CPU
              batch_size,
              epochs,
              H,           # height spectrograms
              W):          # width spectrograms
        
        self.model.train()
        
        inputs = Variable(torch.from_numpy(mixed_input)).to(device)
        inputs = inputs.reshape(n,1,H,W)
        inputs = inputs.type(torch.cuda.FloatTensor)
      
        target = Variable(torch.from_numpy(target)).to(device)
        target = target.reshape(n,1,H,W)
        target = target.type(torch.cuda.FloatTensor)
        
        loss_save = []
        current_epoch = 0
        
        for current_epoch in range(epochs):
            
            idx  = random.sample(list(np.arange(n)),batch_size) # 50 random indices on the 100 specs 
            
            outputs = self.model(inputs[idx,:,:,:])
          
            loss = self.criterion(outputs, target[idx,:,:,:])
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
   
            loss_save.append(loss.item())
    
        # monitor progress
            if current_epoch%500==0:
                print('epoch {}, loss {}'.format(current_epoch, loss.item()))
        
                #plt.imshow(inputs[0,:,:,:].reshape(freq_bins,time_frames).cpu().numpy(),aspect='auto', origin='lower')
                #plt.title('Mixed input')
                #plt.show()
        
                plt.imshow(target[0,:,:].reshape(H,W).cpu().numpy(),aspect='auto', origin='lower', extent=[0,300,0,5])
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Target')
                plt.show()
        
                plt.imshow(outputs[0,:,:,:].reshape(H,W).detach().cpu().numpy(),aspect='auto', origin='lower', extent=[0,300,0,5])
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Separated output')
                plt.show()
          
        plt.plot(loss_save)
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.show()
        
        print('Saving trained model...')
    
        now = datetime.now()
        time_string = now.strftime("%d-%m-%Y_%H:%M:%S")
        torch.save(self.model, self.path + '/' + self.filename + '_' + time_string + '.pt')
        
        print('Model '+ self.filename + '_' + time_string + '.pt' + ' saved.')
