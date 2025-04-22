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
from torch.nn.init import xavier_uniform_

class AdMLP_nge(nn.Module):
    def __init__(self, num_hp=25, num_features=64):
        super(AdMLP_nge, self).__init__()
        self.cmcode = nn.Parameter(torch.Tensor(1,256))       
        self.hpcode = nn.Parameter(torch.Tensor(num_hp,256))

        layers = []
        for i in range(int(np.log2(8))):
            layers.append(nn.Sequential(nn.Conv2d(num_features*2**i, num_features*2**(i+1), kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features*2**(i+1)),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))
        self.extractor = nn.Sequential(*layers)

        #self.wt_c = nn.Parameter(torch.Tensor(1,4096))
        self.fc1 = nn.Linear(512*8*8, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(768, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048, 4096)
        
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

        self._init_pm()

    def _init_pm(self):
        xavier_uniform_(self.cmcode)
        xavier_uniform_(self.hpcode)
        #xavier_uniform_(self.wt_c)

    def forward(self, inputs, num_hp=25):
        bs=inputs.size(0)
        features=self.extractor(inputs)
        features=features.reshape(bs,-1)  
        features=self.fc1(features)
        features=self.bn1(features)
        features=self.relu(features)
        features=self.fc2(features)
        features=self.bn2(features)
        features=self.relu(features)
        features=self.fc3(features)
        cmcode=self.cmcode.repeat(num_hp*bs,1)
        hpcode=self.hpcode.unsqueeze(1).repeat(1,bs,1)
        features=features.unsqueeze(0).repeat(num_hp,1,1)
        code=torch.cat([cmcode.reshape(num_hp,bs,-1),hpcode,features],dim=2)
        y=self.fc4(code.reshape(num_hp*bs,-1))
        y=self.bn3(y)
        y=self.relu(y)
        y=self.fc5(y)
        y=self.sigmoid(y)
        y=y.reshape(num_hp,bs,-1).transpose(0,1)
        y=y.reshape(bs,num_hp*64,64)
        #y=y/y.sum(dim=1,keepdim=True)
        return y

class AdMLP(nn.Module):
    def __init__(self, num_hp=25, num_features=64):
        super(AdMLP, self).__init__()
        self.cmcode = nn.Parameter(torch.Tensor(1,256))       
        self.hpcode = nn.Parameter(torch.Tensor(num_hp,256))

        layers = []
        for i in range(int(np.log2(8))):
            layers.append(nn.Sequential(nn.Conv2d(num_features*2**i, num_features*2**(i+1), kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features*2**(i+1)),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2, padding=0)))
        self.extractor = nn.Sequential(*layers)

        #self.wt_c = nn.Parameter(torch.Tensor(1,4096))
        self.fc1 = nn.Linear(512*8*8, 4096)
        self.bn1 = nn.BatchNorm1d(4096)
        self.fc2 = nn.Linear(4096, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.fc4 = nn.Linear(768, 2048)
        self.bn3 = nn.BatchNorm1d(2048)
        self.fc5 = nn.Linear(2048, 4096)
        
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()

        self._init_pm()

    def _init_pm(self):
        xavier_uniform_(self.cmcode)
        xavier_uniform_(self.hpcode)
        #xavier_uniform_(self.wt_c)

    def forward(self, inputs, num_hp=25):
        bs=inputs.size(0)
        features=self.extractor(inputs)
        features=features.reshape(bs,-1)  
        features=self.fc1(features)
        features=self.bn1(features)
        features=self.relu(features)
        features=self.fc2(features)
        features=self.bn2(features)
        features=self.relu(features)
        features=self.fc3(features)
        cmcode=self.cmcode.repeat(num_hp*bs,1)
        hpcode=self.hpcode.unsqueeze(1).repeat(1,bs,1)
        features=features.unsqueeze(0).repeat(num_hp,1,1)
        code=torch.cat([cmcode.reshape(num_hp,bs,-1),hpcode,features],dim=2)
        y=self.fc4(code.reshape(num_hp*bs,-1))
        y=self.bn3(y)
        y=self.relu(y)
        y=self.fc5(y)
        y=self.sigmoid(y)
        y=y.reshape(num_hp,bs,-1).transpose(0,1)
        y=y.reshape(bs,num_hp*64,64)
        #y=y/y.sum(dim=1,keepdim=True)
        return y
