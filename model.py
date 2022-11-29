from tracemalloc import start
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
        self.percent = 0
        self.location = 0

    def setMutation(self, types):
        self.mutation = types

    def setMutationType(self, types):
        self.mutationType = types

    def setPercent(self, p):
        self.percent = p

    def setLocation(self, l):
        self.location = l

    def __changeWeight(self, x, where, percent):
        pass

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
                    index = torch.tensor([i for i in range(0,row//2)])
                elif mu == 6:
                    index = torch.tensor([i for i in range(row//2, row)])
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
                        index = torch.tensor([i for i in range(0,row//2)])
                    else:
                        index = torch.tensor([i for i in range(row//2, row)])
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

    #n-palace grid remove experiments
    #percent: 3---3*3 grid, n ---- n*n grid. [2,5]
    #location: from top to bottom, from left to right: 0,2,3....,n*n-1
    def __removeGrid__(self, x, percent, location):
        if percent < 2 or percent > 5 or location < 0 or location > percent * percent-1:
            print("error")
            exit(0)
            #execute the threshold, do nothing 
        for instance in x:
            for ind, fea in enumerate(instance):
                row, col = instance[ind].shape
                block_row_count, block_col_count = row//percent, col//percent
                remove_block_row_start, remove_block_col_start = location//percent, location%percent
                start_row = block_row_count * remove_block_row_start
                start_col = block_col_count * remove_block_col_start
                tar = torch.zeros_like(fea).to(fea.device)

                index = [[i if j >= start_row else 0 for i in range(start_col, start_col+block_col_count)] for j in range(0, start_row+block_row_count)]
                #print(index, fea.shape, start_col, block_col_count, start_row, block_row_count)
                #index_col = [i for i in range(0, col-2)]
                # index = torch.tensor([[i for i in range(start_row, start_row+block_row_count) ] for j in range(start_col, start_col+block_col_count)]).to(fea.device)
                # instance[ind] = instance[ind].scatter(1, index, tar)
                # index_row = torch.tensor(index_row)
                # index_col = torch.tensor(index_col)
                # print(index)
                # print(start_row, start_col,start_row+block_row_count+1 ,start_col+block_col_count+1)
                instance[ind] = instance[ind].scatter(1, torch.tensor(index).to(fea.device), tar)
                # print(index)
                # print(instance[ind], start_row, start_col,start_row+block_row_count+1 ,start_col+block_col_count+1)
                # print("!!!!!!!!!!!!!!!!!!!!!!!")
                # print("\r\n\r\n")
                #instance[ind][index_row] = tar[index_row]
                #print(row,col, index_row, index_col, instance[ind][index_row, index_col])
                #print(block_row_count, block_col_count,start_row,start_col,instance[ind][index_row, index_col])
        # for instance in x:
        #     print("\r\n\r\n")
        #     for ind, fea in enumerate(instance):
        #         for j in fea:
                    
        #             print(j)
        #     print("\r\n\r\n")
        return x

    def forward(self, x):
        pass



#inherit

#for mnist model 1
class Net1(Net):
    def __init__(self, ):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 1)
        self.conv2 = nn.Conv2d(6, 16, 5, 1)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.mutation = 0
        self.mutationType = 0
        self.percent = 0
        self.location = 0
    


    def forward(self, x):
        x = self.conv1(x)
        if self.mutationType == 's':
            x = self.__remove(x, self.mutation)
        elif self.mutationType == 'c':
            x = self.__remove1(x, self.mutation)
        elif self.mutationType == 'r':
            x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output



#for mnist model 2
class Net2(Net):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 1)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(1024, 200)
        self.fc2 = nn.Linear(200, 10)
        self.mutation = 0
        self.mutationType = 0
        self.percent = 0
        self.location = 0
    
    def forward(self, x):
        x = self.conv1(x)
        if self.mutationType == 's':
            x = self.__remove(x, self.mutation)
        elif self.mutationType == 'c':
            x = self.__remove1(x, self.mutation)
        elif self.mutationType == 'r':
            x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = self.conv2(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = self.conv4(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



#for cifar
class Net3(Net):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 128, 3)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 10)
        self.mutation = 0
        self.mutationType = 0
        self.percent = 0
        self.location = 0
    
    def forward(self, x):
        x = self.conv1(x)
        if self.mutationType == 's':
            x = self.__remove(x, self.mutation)
        elif self.mutationType == 'c':
            x = self.__remove1(x, self.mutation)
        elif self.mutationType == 'r':
            x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = self.conv2(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = self.conv4(x)
        # if self.mutationType == 's':
        #     x = self.__remove(x, self.mutation)
        # elif self.mutationType == 'c':
        #     x = self.__remove1(x, self.mutation)
        # elif self.mutationType == 'r':
        #     x = self.__removeGrid__(x, self.percent, self.location)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output