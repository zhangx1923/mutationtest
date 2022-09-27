from tomlkit import boolean, integer
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.module):
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

    #no.1 mutation, change weights
    def forward_mu1(self, x):
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
