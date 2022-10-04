import torch
import torch.nn as nn
import torch.nn.functional as F

a = torch.rand(5,5)
l = [[1],[2],[3],[4,1],[0]]
max_length = max([len(i) for i in l])
index = [i + i[-1:]* (max_length-len(i)) for i in l]
t = torch.zeros(5,5)
print(a)
print(t)
print(index)
print(a.scatter(1,torch.tensor(index), t))