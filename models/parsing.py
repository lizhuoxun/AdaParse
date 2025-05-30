##########################
### MODEL
##########################
import torch
import torch.nn as nn

class ParsingNet(torch.nn.Module):

    def __init__(self,num_hidden):
        super(ParsingNet, self).__init__()
        
        ### ENCODER
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(4608, num_hidden)
        
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        
        
        self.fc8_1=nn.Linear(64, 1)
        self.fc8_2=nn.Linear(64, 1)
        self.fc8_3=nn.Linear(64, 1)
        self.fc8_4=nn.Linear(64, 1)
        self.fc8_5=nn.Linear(64, 1)
        self.fc8_6=nn.Linear(64, 1)
        self.fc8_7=nn.Linear(64, 1)
        self.fc8_8=nn.Linear(64, 1)
        self.fc8_9=nn.Linear(64, 1)
        self.fc8_10=nn.Linear(64, 4)
        self.fc8_11=nn.Linear(64, 4)
        self.fc8_12=nn.Linear(64, 4)
        self.fc8_13=nn.Linear(64, 2)
        self.fc8_14=nn.Linear(64, 2)
        self.fc8_15=nn.Linear(64, 2)
        
        
        #self.fc4 = nn.Linear(512, 256)
        #self.fc5 = nn.Linear(256, 64)
        self.fc9_1=nn.Linear(64, 2)
        self.fc9_2=nn.Linear(64, 2)
        self.fc9_3=nn.Linear(64, 2)
        
        
        #self.fc6 = nn.Linear(512, 256)
        #self.fc7 = nn.Linear(256, 64)
        
        self.fc10_1=nn.Linear(64, 2)
        self.fc10_2=nn.Linear(64, 2)
        self.fc10_3=nn.Linear(64, 2)
        self.fc10_4=nn.Linear(64, 2)
        self.fc10_5=nn.Linear(64, 2)
        self.fc10_6=nn.Linear(64, 2)
        self.fc10_7=nn.Linear(64, 2)
        self.fc10_8=nn.Linear(64, 2)
        self.fc10_9=nn.Linear(64, 2)
        self.fc10_10=nn.Linear(64, 2)
        
        self.relu=nn.ReLU()
        self.sigmoid=nn.Sigmoid()
        

    def forward(self, x):
        
        ### ENCODER
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out_512 = self.fc1(out)
        out_512=self.relu(out_512)
        
        out1=self.fc2(out_512)
        out1=self.relu(out1)
        out1=self.fc3(out1)
        out1=self.relu(out1)
        
        #out2=self.fc4(out_512)
        #out2=self.relu(out2)
        #out2=self.fc5(out2)
        #out2=self.relu(out2)
        
        #out3=self.fc6(out_512)
        #out3=self.relu(out3)
        #out3=self.fc7(out3)
        #out3=self.relu(out3)
        
        out1=out1.reshape(25,int(x.size(0)/25),-1)
        outn1_1=self.fc8_1(out1[0])
        outn1_2=self.fc8_2(out1[1])
        outn1_3=self.fc8_3(out1[2])
        outn1_4=self.fc8_4(out1[3])
        outn1_5=self.fc8_5(out1[4])
        outn1_6=self.fc8_6(out1[5])
        outn1_7=self.fc8_7(out1[6])
        outn1_8=self.fc8_8(out1[7])
        outn1_9=self.fc8_9(out1[8])
        outn1=torch.cat([outn1_1,outn1_2,outn1_3,outn1_4,outn1_5,outn1_6,outn1_7,outn1_8,outn1_9],dim=1)
        outn2=self.fc8_10(out1[9])
        outn3=self.fc8_11(out1[10])
        outn4=self.fc8_12(out1[11])
        outn5=self.fc8_13(out1[12])
        outn6=self.fc8_14(out1[13])
        outn7=self.fc8_15(out1[14])
      
        
        outn1=self.sigmoid(outn1)
        outn2=self.sigmoid(outn2)
        outn3=self.sigmoid(outn3)
        outn4=self.sigmoid(outn4)
        outn5=self.sigmoid(outn5)
        outn6=self.sigmoid(outn6)
        outn7=self.sigmoid(outn7)
        
        out3L1=self.fc9_1(out1[15:20].mean(dim=0))
        out3L2=self.fc9_2(out1[20:24].mean(dim=0))
        out3L3=self.fc9_3(out1[24])
        
        out3L1=self.sigmoid(out3L1)
        out3L2=self.sigmoid(out3L2)
        out3L3=self.sigmoid(out3L3)
        
        out9L1=self.fc10_1(out1[15])
        out9L2=self.fc10_2(out1[16])
        out9L3=self.fc10_3(out1[17])
        out9L4=self.fc10_4(out1[18])
        out9L5=self.fc10_5(out1[19])
        out9L6=self.fc10_6(out1[20])
        out9L7=self.fc10_7(out1[21])
        out9L8=self.fc10_8(out1[22])
        out9L9=self.fc10_9(out1[23])
        out9L10=self.fc10_10(out1[24])
        
        out9L1=self.sigmoid(out9L1)
        out9L2=self.sigmoid(out9L2)
        out9L3=self.sigmoid(out9L3)
        out9L4=self.sigmoid(out9L4)
        out9L5=self.sigmoid(out9L5)
        out9L6=self.sigmoid(out9L6)
        out9L7=self.sigmoid(out9L7)
        out9L8=self.sigmoid(out9L8)
        out9L9=self.sigmoid(out9L9)
        out9L10=self.sigmoid(out9L10)
        
        out3L1_1=torch.unsqueeze(out3L1[:,1].clone(),1)
        out3L2_1=torch.unsqueeze(out3L2[:,1].clone(),1)
        out3L3_1=torch.unsqueeze(out3L3[:,1].clone(),1)
        
        outh9L1=out9L1*out3L1_1
        outh9L2=out9L2*out3L1_1
        outh9L3=out9L3*out3L1_1
        outh9L4=out9L4*out3L1_1
        outh9L5=out9L5*out3L1_1
        outh9L6=out9L6*out3L2_1
        outh9L7=out9L7*out3L2_1
        outh9L8=out9L8*out3L2_1
        outh9L9=out9L9*out3L2_1
        outh9L10=out9L10*out3L3_1
        
        
        
        return outn1,outn2,outn3,outn4,outn5,outn6,outn7,out3L1,out3L2,out3L3,outh9L1,outh9L2,outh9L3,outh9L4,outh9L5,outh9L6,outh9L7,outh9L8,outh9L9,outh9L10
