from torchvision import datasets, models, transforms
import os
import torch
from torch.autograd import Variable
from skimage import io
from scipy import fftpack
import numpy as np
from torch import nn
import datetime
from models import parsing
from models import person_fen
from models import encoders
from models import data
import torch.nn.functional as F
import argparse
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--device', default=0, type=int, help='GPU')
parser.add_argument('--data',default='./data_final/',help='root directory for data')
parser.add_argument('--ground_truth_dir',default='./dataset/',help='directory for ground truth')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--savedir', default='./results')
parser.add_argument('--model_dir', default='./results/result_diff/adapt/2024-06-20 13:52:00.718600/model/')
parser.add_argument('--N_given', nargs='+', help='position number of GM from list of GMs used in testing', default=[7,11,12,28,32,35,50,61,62,73,79,80,87,94,106])
parser.add_argument('--test', nargs='+', help='GM used in testing', default=[])

test1=['Z_1','Z_2','Z_3']
test2=['Z_3','Z_4','Z_5']
test3=['Z_2','Z_6','Z_7']
test4=['Z_4','Z_5','Z_7']

opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)
device=torch.device("cuda:%d"%opt.device)
torch.backends.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

data_path=opt.data
save_dir=opt.savedir
if opt.test[0]=='1':
    test_list=test1
elif opt.test[0]=='2':
    test_list=test2
elif opt.test[0]=='3':
    test_list=test3
elif opt.test[0]=='4':
    test_list=test4
else:
    test_list=opt.test

transform_train = transforms.Compose([
transforms.Resize((64,64)),
transforms.ToTensor(),
transforms.Normalize((0.6490, 0.6490, 0.6490), (0.1269, 0.1269, 0.1269))
])

test_set=data.FilterableImageFolder(data_path, transform_train, valid_classes=test_list)
b_s=opt.batch_size
test_loader = torch.utils.data.DataLoader(test_set,batch_size=b_s,shuffle =True, num_workers=1)

model=person_fen.PersonalizedFEN().to(device)
   
model_params = list(model.parameters())    
optimizer = torch.optim.Adam(model_params, lr=opt.lr)

model_2=parsing.ParsingNet(num_hidden=512).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=opt.lr)

l1=torch.nn.L1Loss().to(device)


def validation(batch,labels,g_t_net,g_t_loss):
    model.eval()
    model_2.eval()
    batch_size=len(labels)
    with torch.no_grad():
        y,low_freq_part,max_value,y_orig,hpfingerprints,y_trans,hpfingerprints_gray=model(batch.to(device),device)
        outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10=model_2(hpfingerprints)
    
        outn2=torch.argmax(outn2,-1).to(device)
        outn3=torch.argmax(outn3,-1).to(device)
        outn4=torch.argmax(outn4,-1).to(device)
        outn5=torch.argmax(outn5,-1).to(device)
        outn6=torch.argmax(outn6,-1).to(device)
        outn7=torch.argmax(outn7,-1).to(device)

        out_netd=torch.stack([outn2,outn3,outn4,outn5,outn6,outn7],dim=1)

        out_loss=torch.stack([outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10],dim=1)
        out_loss=torch.argmax(out_loss,-1).to(device)

        #out_d=torch.cat([out_netd,out_loss],dim=1)
    return outn1,out_netd,out_loss

ground_truth_net_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "DM_net_arch.npy"))
ground_truth_loss_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "DM_loss.npy"))

data_set=datasets.ImageFolder(data_path)
N_given=[data_set.class_to_idx[m] for m in test_list]
N_all=list(range(123))
N = [x for x in N_all if x in N_given]
ground_truth_net=ground_truth_net_all[N]
ground_truth_loss=ground_truth_loss_all[N]



print(len(test_set))
print(test_set.class_to_idx)
maxflag=-100
epochs=20

for epoch in range(epochs):
    print("epoch:",epoch)
    if opt.model_dir != '':
        state1 = torch.load(opt.model_dir+"123_model_%d.pickle"%(epoch))
        optimizer.load_state_dict(state1['optimizer_1'])
        model.load_state_dict(state1['state_dict_cnn'])
        optimizer_2.load_state_dict(state1['optimizer_2'])
        model_2.load_state_dict(state1['state_dict_class'])

    l1loss=[]

    for batch_idx, (inputs,labels) in enumerate(test_loader):
        if batch_idx%100==0: print("batch_index:",batch_idx)
        g_t_net_batch= torch.empty(labels.shape[0], 15)
        g_t_loss_batch= torch.empty(labels.shape[0], 10)
        for i in range(labels.shape[0]):
            g_t_net_batch[i,:]=ground_truth_net[labels[i]]
            g_t_loss_batch[i,:]=ground_truth_loss[labels[i]]
        
        out_netc_batch,out_netd_batch,out_loss_batch=validation(Variable(torch.FloatTensor(inputs)),Variable(torch.LongTensor(labels)),Variable(torch.FloatTensor(g_t_net_batch)),Variable(torch.FloatTensor(g_t_loss_batch)))
        g_t_netc_batch=g_t_net_batch[:,0:9]
        g_t_netd_batch=g_t_net_batch[:,9:15]      

        if batch_idx==0:
            out_netc=out_netc_batch.detach()
            gt_netc=g_t_netc_batch.detach()
            out_netd=out_netd_batch.detach()
            gt_netd=g_t_netd_batch.detach()
            out_loss=out_loss_batch.detach()
            gt_loss=g_t_loss_batch.detach()
        else:
            out_netc=torch.cat([out_netc,out_netc_batch.detach()],dim=0)
            gt_netc=torch.cat([gt_netc,g_t_netc_batch.detach()],dim=0)
            out_netd=torch.cat([out_netd,out_netd_batch.detach()],dim=0)
            gt_netd=torch.cat([gt_netd,g_t_netd_batch.detach()],dim=0)
            out_loss=torch.cat([out_loss,out_loss_batch.detach()],dim=0)
            gt_loss=torch.cat([gt_loss,g_t_loss_batch.detach()],dim=0)
    l1loss=np.zeros(9)
    for i in range(9):
        l1loss[i]=l1(out_netc[:,i].to(device), gt_netc[:,i].to(device)).item()
    
    out_netd=np.asarray(out_netd.cpu())
    gt_netd=np.asarray(gt_netd.cpu())
    out_loss=np.asarray(out_loss.cpu())
    gt_loss=np.asarray(gt_loss.cpu())
    acc_netd=np.zeros(6)
    f1_netd=np.zeros(6)
    acc_loss=np.zeros(10)
    f1_loss=np.zeros(10)
    for i in range(6):
        acc_netd[i]=accuracy_score(gt_netd[:,i], out_netd[:,i])
        f1_netd[i]=f1_score(gt_netd[:,i], out_netd[:,i], average='macro',zero_division=0)
    for i in range(10):
        acc_loss[i]=accuracy_score(gt_loss[:,i], out_loss[:,i])
        f1_loss[i]=f1_score(gt_loss[:,i], out_loss[:,i], average='macro',zero_division=0)
    Accuracy_net=np.mean(acc_netd)
    F1_net=np.mean(f1_netd)
    Accuracy_loss=np.mean(acc_loss)
    F1_loss=np.mean(f1_loss)
    print("L1loss:",l1loss.mean(),"Standard deviation:",l1loss.std())
    print("Accuracy(net):",Accuracy_net)
    print("F1 score(net):",F1_net)
    print("Accuracy(loss):",Accuracy_loss)
    print("F1 score(loss):",F1_loss)
    flag=F1_net+F1_loss-5*l1loss.mean()
    if epoch==0 or flag>maxflag:
        maxflag=flag
        result=[epoch,l1loss.mean(),F1_net,F1_loss]

print("Best result:",result)
 
    
