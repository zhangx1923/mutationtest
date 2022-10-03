import torch
import torch.nn as nn
import torch.nn.functional as F

#CNN中的神经元：n个卷积核，每个卷积核卷一次生成一个神经元
#卷积核3*3*channel，对应一个bias
#https://cs231n.github.io/convolutional-networks/ local Connectivity 一节
#mutation types:
#1. change weights
#2. change bias
#3. remove neurons  3.1!! conv layer   3.2 fc layer

#逐个删除神经元，生成图片
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.mutation = 0

    def setMutation(self, types):
        self.mutation = types

    def forward_mu1(self, x):
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        for ins in x:
            for i,fea in enumerate(ins):
                #each feature map
                for m in range(len(fea)):
                    for n in range(len(fea[m])):
                        print(fea[m][n])
                        fea[m][n] = 0
                        print(fea[m][n])
                        print("!!!")


        output = F.log_softmax(x, dim=1)
        return output

    def forward(self, x):
        if self.mutation == 0:
            x = self.conv1(x)
            x = F.relu(x)
            x = self.conv2(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)
            x = self.dropout1(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc2(x)
            output = F.log_softmax(x, dim=1)
            return output
        elif self.mutation == 1:
            self.forward_mu1(x)
