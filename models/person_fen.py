from torchvision import datasets, models, transforms
import os
import torch
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics
import argparse
from models.encoders import DualEncoder

try:
    from torch import irfft
    from torch import rfft
except ImportError:
    from torch.fft import ifft2
    from torch.fft import fft2
    def rfft(x, signal_ndim):
        t = fft2(x, dim = (-signal_ndim, -1))
        return torch.stack((t.real, t.imag), -1)
    def irfft(x, d, signal_sizes):
        return ifft2(torch.complex(x[:,:,0], x[:,:,1]), s = signal_sizes, dim = (-d))

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0,n,None) 
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n,None,None)
                  for i in range(X.dim()))
    #print(axis,n,f_idx,b_idx)
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front],axis)

def fftshift(real, imag):
    for dim in range(1, len(real.size())):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return real, imag


class PersonalizedFEN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(PersonalizedFEN, self).__init__()
        self.hypernet = DualEncoder()
        num_mid_layers = 5
        num_adapt_layers = 5

        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - num_adapt_layers - num_mid_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        self.layers = nn.Sequential(*layers)
        
        layers1 = []
        for i in range(num_mid_layers):
            layers1.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        self.layers1 = nn.Sequential(*layers1)

        layers2 = []
        for i in range(num_adapt_layers):
            layers2.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers2.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers2 = nn.Sequential(*layers2)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs, device, num_hp=25):
        y = self.layers(inputs)
        w = self.hypernet(y)
        y = self.layers1(y)
        ys = y.size()
        y = y.permute(0,2,3,1)
        residual = torch.zeros(ys[0],ys[2],ys[3],num_hp*ys[1]).to(device)
        for i in range(ys[0]):
            residual[i] = F.linear(y[i], weight=w[i])
        residual = residual.reshape(ys[0],ys[2],ys[3],num_hp,ys[1])
        residual = residual.permute(3,0,4,1,2)
        residual = self.layers2(residual.reshape(num_hp*ys[0],ys[1],ys[2],ys[3]))
        residual_1 = residual.clone()
        
        residual_gray=0.299*residual_1[:,0,:,:].clone()+0.587*residual_1[:,1,:,:].clone()+0.114*residual_1[:,2,:,:].clone()
        
        thirdPart_fft_1=rfft(residual_gray, signal_ndim=2)
        
        thirdPart_fft_1_orig=thirdPart_fft_1.clone()
        
        thirdPart_fft_1[:,:,:,0],thirdPart_fft_1[:,:,:,1]=fftshift(thirdPart_fft_1[:,:,:,0],thirdPart_fft_1[:,:,:,1])
        thirdPart_fft_1=torch.sqrt(torch.clamp(thirdPart_fft_1[:,:,:,0]**2+thirdPart_fft_1[:,:,:,1]**2,min=1e-4))
        n=25
        (_,w,h)=thirdPart_fft_1.shape
        half_w, half_h = int(w/2), int(h/2)
        thirdPart_fft_2=thirdPart_fft_1[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1].clone()
        thirdPart_fft_3=thirdPart_fft_1.clone()
        thirdPart_fft_3[:,half_w-n:half_w+n+1,half_h-n:half_h+n+1]=0
        max_value=torch.max(thirdPart_fft_3)
        thirdPart_fft_4=thirdPart_fft_1.clone()
        thirdPart_fft_4=torch.transpose(thirdPart_fft_4,1,2)
        return thirdPart_fft_1,thirdPart_fft_2, max_value, thirdPart_fft_1_orig,residual, thirdPart_fft_4, residual_gray
