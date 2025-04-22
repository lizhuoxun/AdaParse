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
from models import data
import torch.nn.functional as F
import argparse
from torch import autograd

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


#################################################################################################################
# HYPER PARAMETERS INITIALIZING
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--wt_decay', default=0.0001, type=float, help='weight decay')
parser.add_argument('--device', default=0, type=int, help='GPU')
parser.add_argument('--data',default='./data_final/',help='root directory for data')
parser.add_argument('--ground_truth_dir',default='./dataset/',help='directory for ground truth')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--batch_size', default=16, type=int, help='batch size')
parser.add_argument('--savedir', default='./results/ablation/weightgen/iterative')
parser.add_argument('--model_dir', default='')
parser.add_argument('--N_given', nargs='+', help='position number of GM from list of GMs used in testing', default=[7,11,12,28,32,35,50,61,62,73,79,80,87,94,106])
parser.add_argument('--test', nargs='+', help='GM used in testing', default=['AAE','ADAGAN_C','BEGAN','BETA_H','BIGGAN_256','COCO_GAN','CRAMER_GAN','DEEPFOOL','DRIT','FASTPIXEL','FVBN','SRFLOW'])

test1=['ADV_FACES','BETA_B','BETA_TCVAE','BIGGAN_128','DAGAN_C','DRGAN','FGAN','PIXELCNN','PIXELCNN_PP','RSGAN_HALF','STARGAN','VAE_GAN']
test2=['AAE','ADAGAN_C','BEGAN','BETA_H','BIGGAN_256','COCO_GAN','CRAMER_GAN','DEEPFOOL','DRIT','FASTPIXEL','FVBN','SRFLOW']
test3=['BICYCLE_GAN','BIGGAN_512','CRGAN_C','FACTOR_VAE','FGSM','ICRGAN_C','LOGAN','MUNIT','PIXELSNAIL','STARGAN_2','SURVAE_FLOW_MAXPOOL','VAE_FIELD']
test4=['GFLM','IMAGE_GPT','LSGAN','MADE','PIX2PIX','PROG_GAN','RSGAN_REG','SEAN','STYLEGAN','SURVAE_FLOW_NONPOOL','WGAN_DRA','YLG']

#clip_value=0.01

opt = parser.parse_args()
print(opt)
print("Random Seed: ", opt.seed)

device=torch.device("cuda:%d"%opt.device)
torch.backends.deterministic = True
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)

sig = str(datetime.datetime.now())
  
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

ground_truth_net_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_net_116_15dim.npy"))
ground_truth_loss_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_loss_116_3dim.npy"))
ground_truth_loss_9_all=torch.from_numpy(np.load(opt.ground_truth_dir+ "ground_truth_loss_116_9dim.npy"))

data_set=datasets.ImageFolder(data_path)
train_list=[m for m in data_set.classes if m not in test_list]
N_given=[data_set.class_to_idx[m] for m in test_list]
N_all=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115]
N = [x for x in N_all if x not in N_given]
ground_truth_net=ground_truth_net_all[N]
ground_truth_loss=ground_truth_loss_all[N]
ground_truth_loss_9=ground_truth_loss_9_all[N]


os.makedirs('%s/%s' % (save_dir, sig), exist_ok=True)
os.makedirs('%s/%s/model' % (save_dir, sig), exist_ok=True)


transform_train = transforms.Compose([
transforms.Resize((64,64)),
transforms.ToTensor(),
transforms.Normalize((0.6490, 0.6490, 0.6490), (0.1269, 0.1269, 0.1269))
])


train_set=data.FilterableImageFolder(data_path, transform_train, valid_classes=train_list)
test_set=data.FilterableImageFolder(data_path, transform_train, valid_classes=test_list)
b_s=opt.batch_size
train_loader = torch.utils.data.DataLoader(train_set,batch_size=b_s,shuffle =True, num_workers=1)
test_loader = torch.utils.data.DataLoader(test_set,batch_size=b_s,shuffle =True, num_workers=1)

model=person_fen.PersonalizedFEN().to(device)    
optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wt_decay)

model_2=parsing.ParsingNet(num_hidden=512).to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=opt.lr, weight_decay=opt.wt_decay)

weightn1 = torch.tensor([116/72, 116/19, 116/19,116/6])
weightn2 = torch.tensor([116/10, 116/74, 116/8,116/24])
weightn3 = torch.tensor([116/15, 116/74, 116/24,116/3])
weightn4 = torch.tensor([116/80, 116/36])
weightn5 = torch.tensor([116/39, 116/77])
weightn6 = torch.tensor([116/67, 116/49])

weight3L1 = torch.tensor([116/32, 116/84])
weight3L2 = torch.tensor([116/75, 116/41])
weight3L3 = torch.tensor([116/67, 116/49])

weight9L1 = torch.tensor([116/71, 116/45])
weight9L2 = torch.tensor([116/95, 116/21])
weight9L3 = torch.tensor([116/102, 116/14])
weight9L4 = torch.tensor([116/112, 116/4])
weight9L5 = torch.tensor([116/71, 116/45])
weight9L6 = torch.tensor([116/102, 116/14])
weight9L7 = torch.tensor([116/102, 116/14])
weight9L8 = torch.tensor([116/110, 116/6])
weight9L9 = torch.tensor([116/105, 116/11])
weight9L10 = torch.tensor([116/67, 116/49])



l1=torch.nn.MSELoss().to(device)

l_cn1 = torch.nn.CrossEntropyLoss(weightn1).to(device)
l_cn2 = torch.nn.CrossEntropyLoss(weightn2).to(device)
l_cn3 = torch.nn.CrossEntropyLoss(weightn3).to(device)
l_cn4 = torch.nn.CrossEntropyLoss(weightn4).to(device)
l_cn5 = torch.nn.CrossEntropyLoss(weightn5).to(device)
l_cn6 = torch.nn.CrossEntropyLoss(weightn6).to(device)


l_c3L1 = torch.nn.CrossEntropyLoss(weight3L1).to(device)
l_c3L2 = torch.nn.CrossEntropyLoss(weight3L2).to(device)
l_c3L3 = torch.nn.CrossEntropyLoss(weight3L3).to(device)

l_c9L1 = torch.nn.CrossEntropyLoss(weight9L1).to(device)
l_c9L2 = torch.nn.CrossEntropyLoss(weight9L2).to(device)
l_c9L3 = torch.nn.CrossEntropyLoss(weight9L3).to(device)
l_c9L4 = torch.nn.CrossEntropyLoss(weight9L4).to(device)
l_c9L5 = torch.nn.CrossEntropyLoss(weight9L5).to(device)
l_c9L6 = torch.nn.CrossEntropyLoss(weight9L6).to(device)
l_c9L7 = torch.nn.CrossEntropyLoss(weight9L7).to(device)
l_c9L8 = torch.nn.CrossEntropyLoss(weight9L8).to(device)
l_c9L9 = torch.nn.CrossEntropyLoss(weight9L9).to(device)
l_c9L10 = torch.nn.CrossEntropyLoss(weight9L10).to(device)


state = {
    'state_dict_cnn':model.state_dict(),
    'optimizer_1': optimizer.state_dict(),
    'state_dict_class':model_2.state_dict(),
    'optimizer_2': optimizer_2.state_dict()  
}

if opt.model_dir != '':
    state1 = torch.load(opt.model_dir)
    optimizer.load_state_dict(state1['optimizer_1'])
    model.load_state_dict(state1['state_dict_cnn'])
    optimizer_2.load_state_dict(state1['optimizer_2'])
    model_2.load_state_dict(state1['state_dict_class'])


def train(batch, labels,g_t_net,g_t_loss, g_t_loss_9):
    model.train()
    model_2.train()

    y,low_freq_part,max_value,y_orig,hpfingerprints,y_trans,hpfingerprints_gray=model(batch.to(device),device)

    outn1,outn2,outn3,outn4, outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10=model_2(hpfingerprints)
   
    
    n=25
    th=10000*torch.ones(1).to(device)
    zero=torch.zeros(hpfingerprints_gray.shape, dtype=torch.float32).to(device)
    zero1=torch.zeros([low_freq_part.shape[0],2*n+1,2*n+1], dtype=torch.float32).to(device)  

    loss1=0.1*l1(hpfingerprints_gray,zero).to(device)
    loss2=0.05*l1(low_freq_part,zero1).to(device)
    loss3=-0.001*torch.min(torch.cat([max_value.unsqueeze(0),th])).to(device) 
    loss4=l1(y,y_trans).to(device)
    
    loss5_1=10*l1(outn1,g_t_net[:,0:9].to(device))
    
    
    loss5_2=10*l_cn1(outn2,g_t_net[:,9].type(torch.LongTensor).to(device))
    loss5_3=10*l_cn2(outn3,g_t_net[:,10].type(torch.LongTensor).to(device))
    loss5_4=10*l_cn3(outn4,g_t_net[:,11].type(torch.LongTensor).to(device))
    loss5_5=10*l_cn4(outn5,g_t_net[:,12].type(torch.LongTensor).to(device))
    loss5_6=10*l_cn5(outn6,g_t_net[:,13].type(torch.LongTensor).to(device))
    loss5_7=10*l_cn6(outn7,g_t_net[:,14].type(torch.LongTensor).to(device))
    

    loss6_1=10*l_c3L1(out3L1,g_t_loss[:,0].type(torch.LongTensor).to(device))
    loss6_2=10*l_c3L2(out3L2,g_t_loss[:,1].type(torch.LongTensor).to(device))
    loss6_3=10*l_c3L3(out3L3,g_t_loss[:,2].type(torch.LongTensor).to(device))
    

    
    loss7_1=10*l_c9L1(outh9L1,g_t_loss_9[:,0].type(torch.LongTensor).to(device))
    loss7_2=10*l_c9L2(outh9L2,g_t_loss_9[:,1].type(torch.LongTensor).to(device))
    loss7_3=10*l_c9L3(outh9L3,g_t_loss_9[:,2].type(torch.LongTensor).to(device))
    loss7_4=10*l_c9L4(outh9L4,g_t_loss_9[:,3].type(torch.LongTensor).to(device))
    loss7_5=10*l_c9L5(outh9L5,g_t_loss_9[:,4].type(torch.LongTensor).to(device))
    loss7_6=10*l_c9L6(outh9L6,g_t_loss_9[:,5].type(torch.LongTensor).to(device))
    loss7_7=10*l_c9L7(outh9L7,g_t_loss_9[:,6].type(torch.LongTensor).to(device))
    loss7_8=10*l_c9L8(outh9L8,g_t_loss_9[:,7].type(torch.LongTensor).to(device))
    loss7_9=10*l_c9L9(outh9L9,g_t_loss_9[:,8].type(torch.LongTensor).to(device))
    loss7_10=10*l_c9L10(outh9L10,g_t_loss_9[:,9].type(torch.LongTensor).to(device))
    
    
    
    print("loss1:",loss1.item()," loss2:",loss2.item()," loss3:",loss3.item()," loss4:",loss4.item())
    print("loss5_1:",loss5_1.item()," loss 5_2:",loss5_2.item()," loss5_3:",loss5_3.item(),
          " loss5_4:",loss5_4.item()," loss5_5:",loss5_5.item()," loss5_6:",loss5_6.item()," loss5_7:",loss5_7.item())
    print("loss6_1:",loss6_1.item()," loss6_2:",loss6_2.item()," loss6_3:",loss6_3.item())
    print("loss7_1:",loss7_1.item()," loss7_2:",loss7_2.item()," loss7_3:",loss7_3.item()," loss7_4:",loss7_4.item(), " loss7_5:",loss7_5.item()," loss7_6:",loss7_6.item()," loss7_7:",loss7_7.item()," loss7_8:",loss7_8.item()," loss7_9:",loss7_9.item()," loss7_10:",loss7_10.item())
    loss=(loss1+loss2+loss3+loss4+loss5_1+loss5_2+loss5_3+loss5_4+loss5_5+loss5_6+loss5_7+loss6_1+loss6_2+loss6_3+loss7_1+loss7_2+loss7_3+loss7_4+loss7_5+loss7_6+loss7_7+loss7_8+loss7_9+loss7_10)
    print("loss_total:",loss.item())
    optimizer.zero_grad()
    optimizer_2.zero_grad()
    loss.backward()
    optimizer.step()
    optimizer_2.step()
    
    return y, loss.item(), y_orig,hpfingerprints, outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10


print(len(train_set))
print(len(test_set))
print(train_set.class_to_idx)
print(test_set.class_to_idx)

epochs=20


for epoch in range(epochs):
    all_y=[]
    flag=0
    flag1=0
    
    count=0
    number=0
    #with autograd.detect_anomaly():
    for batch_idx, (inputs,labels) in enumerate(train_loader):
        
        g_t_loss_batch= torch.empty(labels.shape[0], 3)
        g_t_loss_batch_9= torch.empty(labels.shape[0], 10)
        g_t_net_batch= torch.empty(labels.shape[0], 15)
        for i in range(labels.shape[0]):
            g_t_net_batch[i,:]=ground_truth_net[labels[i]]
            g_t_loss_batch[i,:]=ground_truth_loss[labels[i]]
            g_t_loss_batch_9[i,:]=ground_truth_loss_9[labels[i]]
        
        out,loss, out_orig,residual,outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1, outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10=train(Variable(torch.FloatTensor(inputs)),Variable(torch.LongTensor(labels)), Variable(torch.FloatTensor(g_t_net_batch)),Variable(torch.FloatTensor(g_t_loss_batch)),Variable(torch.FloatTensor(g_t_loss_batch_9)))
        
        count+=1
        
  
    state = {
        'state_dict_cnn':model.state_dict(),
        'state_dict_class':model_2.state_dict()
    }
    torch.save(state, '%s/%s/model/116_model_%d.pickle' % (save_dir, sig, epoch))
 
    
