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
        self.dropout3 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.mutation = 0
        self.mutationType = 0

    def setMutation(self, types):
        self.mutation = types

    def setMutationType(self, types):
        self.mutationType = types

    def __remove(self, x, mu):
        #1.take one from every two neurons (first row from the beginning)
        #2.take one from every two neurons (first row from the second neuron)
        #3.left top side of diagonal
        #4.right bottom side of diagonal
        #5.top
        #6.bottom
        #7.remove this feature map
        #8.keep this feature map
        #9.right top side of diagonal
        #10.left bottom side of diagonal
        #11. left
        #12. right
        # 13 dropout
        if mu == 0 or mu == 8:
            #no mutation test for this case
            return x
        
        if mu == 13:
            return self.dropout3(x)
        for ins in x:
            for ind,fea in enumerate(ins):    
                #each feature map
                #assign tar's (i,j) value to (i,j) position of fea
                row, col = ins[ind].shape
                if mu == 1:
                    index = torch.tensor([[i for i in range(j%2, col, 2) ] for j in range(row)]).to(fea.device)
                elif mu == 2:
                    index = torch.tensor([[i for i in range(j%2, col, 2) ] for j in range(1,row+1)]).to(fea.device)
                elif mu == 3:
                    index = [[i for i in range(0, row-j, 1) ] for j in range(row)]
                    max_len = max([len(l) for l in index])
                    index = [l + l[-1:] * (max_len - len(l)) for l in index]
                    index = torch.tensor(index).to(fea.device)
                elif mu == 4:
                    index = [[i for i in range(col-1, row-j-2, -1) ] for j in range(row)]
                    max_len = max([len(l) for l in index])
                    index = [l + l[-1:] * (max_len - len(l)) for l in index]
                    index = torch.tensor(index).to(fea.device)
                elif mu == 5:
                    #index = torch.tensor([[i for i in range(0, col, 1) ] for j in range(row//2)]).to(fea.device)
                    index = torch.tensor([i for i in range(0,col//2)])
                elif mu == 6:
                    index = torch.tensor([i for i in range(col//2, col)])
                    # index0 = torch.tensor([[] for j in range(row-row//2)]).to(fea.device)
                    # index1 = torch.tensor([[i for i in range(0, col, 1) ] for j in range(row//2)]).to(fea.device)
                    # index = torch.cat((index0, index1),0)
                elif mu == 7:
                    index = torch.tensor([[i for i in range(0, col, 1) ] for j in range(row)]).to(fea.device)
                elif mu == 9:
                    index = [[i for i in range(j, col, 1) ] for j in range(row)]
                    max_len = max([len(l) for l in index])
                    index = [l + l[-1:] * (max_len - len(l)) for l in index]
                    index = torch.tensor(index).to(fea.device)
                elif mu == 10:
                    index = [[i for i in range(0, j+1, 1) ] for j in range(row)]
                    max_len = max([len(l) for l in index])
                    index = [l + l[-1:] * (max_len - len(l)) for l in index]
                    index = torch.tensor(index).to(fea.device)
                elif mu == 11:
                    index = torch.tensor([[i for i in range(0, col//2, 1) ] for j in range(row)]).to(fea.device)
                elif mu == 12:
                    index = torch.tensor([[i for i in range(col//2, col, 1) ] for j in range(row)]).to(fea.device)
                else:
                    index = []
                tar = torch.zeros_like(fea).to(fea.device)
                if mu == 5 or mu == 6:
                    ins[ind][index] = tar[index]
                else: 
                    ins[ind] = ins[ind].scatter(1, index, tar)
        return x
        # for ins in x:
        #     for i,fea in enumerate(ins):
        #         for m in range(len(fea)):
        #             for n in range(len(fea[m])):
        #                 if mu == 1:
        #                     if (m+n) % 2 != 0:
        #                         fea[m][n] = 0
        #                 elif mu == 2:
        #                     if (m+n) % 2 == 0:
        #                         fea[m][n] = 0                           
        #                 elif mu == 3:
        #                     if m >= n:
        #                        fea[m][n] = 0  
        #                 elif mu == 4:
        #                     if m < n:
        #                         fea[m][n] = 0 
        #                 elif mu == 5:
        #                     if m <= len(fea) // 2:
        #                         fea[m][n] = 0 
        #                 elif mu == 6:
        #                     if m > len(fea) // 2:
        #                         fea[m][n] = 0 
        #                 elif mu == 7:
        #                     fea[m][n] = 0 
        #                 elif mu == 8:
        #                     pass
        #                 elif mu == 9:
        #                     if n >= m:
        #                         fea[m][n] = 0 
        #                 elif mu == 10:
        #                     if n < m:
        #                         fea[m][n] = 0 
        #                 elif mu == 11:
        #                     if n <= len(fea[m]) // 2:
        #                         fea[m][n] = 0 
        #                 elif mu == 12:
        #                     if n > len(fea[m]) // 2:
        #                         fea[m][n] = 0 
        

    def __remove1(self, x, mu):
        #combine the following:
        #1.1.take one from every two neurons (first row from the beginning)
        #1.2.take one from every two neurons (first row from the second neuron)
        #2.1.left top side of diagonal
        #2.2.right bottom side of diagonal
        #3.1.top
        #3.2.bottom
        #4.1.remove this feature map
        #4.2.keep this feature map
        #5.1.right top side of diagonal
        #5.2.left bottom side of diagonal
        #6.1. left
        #6.2. right
        #7 dropout self
        if mu == 0:
            #no mutation test for this case
            return x
        
        if mu == 7:
            return self.dropout3(x)
        
        for ins in x:
            for ind,fea in enumerate(ins): 
                row, col = ins[ind].shape
                if mu == 1:
                    if ind % 2 == 0:
                        index = torch.tensor([[i for i in range(j%2, col, 2) ] for j in range(row)]).to(fea.device)
                    else:
                        index = torch.tensor([[i for i in range(j%2, col, 2) ] for j in range(1,row+1)]).to(fea.device)
                elif mu == 2:
                    if ind % 2 == 0:
                        index = [[i for i in range(0, row-j, 1) ] for j in range(row)]
                        max_len = max([len(l) for l in index])
                        index = [l + l[-1:] * (max_len - len(l)) for l in index]
                        index = torch.tensor(index).to(fea.device)
                    else:
                        index = [[i for i in range(col-1, row-j-2, -1) ] for j in range(row)]
                        max_len = max([len(l) for l in index])
                        index = [l + l[-1:] * (max_len - len(l)) for l in index]
                        index = torch.tensor(index).to(fea.device)                        
                elif mu == 3:
                    if ind % 2 == 0:
                        index = torch.tensor([i for i in range(0,col//2)])
                    else:
                        index = torch.tensor([i for i in range(col//2, col)])
                elif mu == 4:
                    if ind % 2 == 0:
                        index = torch.tensor([[i for i in range(0, col, 1) ] for j in range(row)]).to(fea.device)
                    else:
                        continue
                elif mu == 5:
                    if ind % 2 == 0:
                        index = [[i for i in range(j, col, 1) ] for j in range(row)]
                        max_len = max([len(l) for l in index])
                        index = [l + l[-1:] * (max_len - len(l)) for l in index]
                        index = torch.tensor(index).to(fea.device)
                    else:
                        index = [[i for i in range(0, j+1, 1) ] for j in range(row)]
                        max_len = max([len(l) for l in index])
                        index = [l + l[-1:] * (max_len - len(l)) for l in index]
                        index = torch.tensor(index).to(fea.device)
                elif mu == 6:
                    if ind % 2 == 0:
                        index = torch.tensor([[i for i in range(0, col//2, 1) ] for j in range(row)]).to(fea.device)
                    else:
                        index = torch.tensor([[i for i in range(col//2, col, 1) ] for j in range(row)]).to(fea.device)
                else:
                    index = []
                tar = torch.zeros_like(fea).to(fea.device)
                if mu == 3:
                    ins[ind][index] = tar[index]
                else:
                    ins[ind] = ins[ind].scatter(1, index, tar)
        return x
        
        # for ins in x:
        #     for i,fea in enumerate(ins):
        #         #each feature map
        #         if i % 2 == 0:
        #             for m in range(len(fea)):
        #                 for n in range(len(fea[m])):
        #                     if mu == 1:
        #                         if (m+n) % 2 != 0:
        #                             fea[m][n] = 0                         
        #                     elif mu == 2:
        #                         if m >= n:
        #                             fea[m][n] = 0  
        #                     elif mu == 3:
        #                         if m <= len(fea) // 2:
        #                             fea[m][n] = 0 
        #                     elif mu == 4:
        #                         fea[m][n] = 0
        #                     elif mu == 5:
        #                         if n >= m:
        #                             fea[m][n] = 0 
        #                     elif mu == 6:
        #                         if n <= len(fea[m]) // 2:
        #                             fea[m][n] = 0                 
        #         else:
        #             for m in range(len(fea)):
        #                 for n in range(len(fea[m])):
        #                     if mu == 1:
        #                         if i%2 != 0:
        #                             fea[m][n] = 0                          
        #                     elif mu == 2:
        #                         if m<n:
        #                             fea[m][n] = 0 
        #                     elif mu == 3:
        #                         if m > len(fea) // 2:
        #                             fea[m][n] = 0 
        #                     elif mu == 4:
        #                         pass
        #                     elif mu == 5:
        #                         if n < m:
        #                             fea[m][n] = 0 
        #                     elif mu == 6:
        #                         if n > len(fea[m]) // 2:
        #                             fea[m][n] = 0 
        


    def forward(self, x):
        x = self.conv1(x)
        #x = self.__remove(x, self.mutation)
        # print(self.mutationType)
        if self.mutationType == 's':
            x = self.__remove(x, self.mutation)
        elif self.mutationType == 'c':
            x = self.__remove1(x, self.mutation)
        x = F.relu(x)
        x = self.conv2(x)
        if self.mutationType == 's':
            x = self.__remove(x, self.mutation)
        elif self.mutationType == 'c':
            x = self.__remove1(x, self.mutation)
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
