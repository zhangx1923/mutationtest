import math
import numpy as np
import copy
from itertools import chain



def mo2fea(a, ac, n):
    for i in range(n):
        for j in range(n):
            a[i][j] = ac-a[i][j] if ac-a[i][j] > 0 else 0
    return a

def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def calNorm(a):
    sum = 0
    for row in a:
        for j in row:
            sum += j
    for i in range(0,len(a)):
        for j in range(0,len(a[i])):
            a[i][j] = a[i][j] / sum
    return a

def calScore(a,abefore, f, n, alpha, beta, ac):
    s = 0
    sum_f = 0
    sum_a = 0
    sum_af = 0
    sum_aaa = 0
    avg = 1/(n*n)
    total = n*n
    for i in range(n):
        for j in range(n):
            sum_f += (f[i][j] - avg) * (f[i][j] - avg)
            sum_a += (a[i][j] - avg) * (a[i][j] - avg)
            sum_af += (f[i][j] - a[i][j]) * (f[i][j] - a[i][j])
            sum_aaa += (ac/100-abefore[i][j]/100) * (ac/100-abefore[i][j]/100)
    robust = math.sqrt(sum_f)/total + math.sqrt(sum_a)/total+ math.sqrt(sum_aaa)/total
    print(math.sqrt(sum_f),math.sqrt(sum_a),math.sqrt(sum_aaa), abefore)
    # print(math.sqrt(sum_f), math.sqrt(sum_a))
    #robust = sigmoid(robust)
    acc = ac/100-(math.sqrt(sum_af)/total )
    #acc = sigmoid(acc)
    s = alpha*robust + beta*acc
    return s, robust, acc


def calScore1(a,abefore, f, n, alpha, beta, ac):
    def BCE(y_true, y_pred):
        y_true = np.array(y_true).reshape(-1, 1)
        y_pred = np.array(y_pred).reshape(-1, 1)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        return -np.mean(term_0+term_1, axis=0)[0]
    y1 = list(chain.from_iterable(f))
    y2 = list(chain.from_iterable(a))
    y3 = list(chain.from_iterable(np.array(abefore)/100))
    y4 = [ac/100]*(n*n)
    y5 = [1/(n*n)]*(n*n)
    robust = BCE(y5,y1) + BCE(y5,y2) + BCE(y4,y3)
    acc = ac/100 - BCE(y1,y2)
    s = alpha*robust + beta*acc
    return s, robust, acc

#[0.1, 0.2, 0.25, 0.5, 0.75, 1.0, robust]
mnist1_1_it = [[77.74,83.31],[88.14,73.98]]
mnist1_1_mo = [[86.05,76.45],[74.70,81.34]]
mnist1_1_acc = 94.82
#9840

mnist1_2_it = [[90.98,92.62],[93.67,89.83]]
mnist1_2_mo = [[89.54,71.36],[71.95,86.6]]
mnist1_2_acc = 97.65

mnist1_3_it = [[93.46,94.41],[95.21,93.81]]
mnist1_3_mo = [[85.2,79.85],[73.3,79.75]]
mnist1_3_acc = 97.34

mnist1_4_it = [[93.45,94.93],[95.17,94.56]]
mnist1_4_mo = [[67.48,66.68],[64.34,72.15]]
mnist1_4_acc = 95.42

mnist1_5_it = [[85.81,85.48],[89.48,90.35]]
mnist1_5_mo = [[67.01,63.45],[64.44,66.23]]
mnist1_5_acc = 90.82


mnist1_6_it = [[44.77,46.43],[46.24,48.52]]
mnist1_6_mo = [[62.82,65.90],[62.66,63.17]]
mnist1_6_acc = 88.69

mnist1_7_it = [[],[],[]]
mnist1_7_mo = [[],[],[]]
mnist1_7_acc = 0




mnist2_1_it = [[72.78,87.97],[91.05,79.73]]
mnist2_1_mo = [[91.91,85.43],[73.32,94.75]]
mnist2_1_acc = 99.12

mnist2_2_it = [[95.97,96.43],[97.46,95.99]]
mnist2_2_mo = [[86.04,85.15],[76.17,91.64]]
mnist2_2_acc = 99.04

mnist2_3_it = [[97.68,97.60],[98.06,97.72]]
mnist2_3_mo = [[82.99,84.54],[75.68,88.69]]
mnist2_3_acc = 98.83

mnist2_4_it = [[98.33,98.18],[98.40,98.14]]
mnist2_4_mo = [[76.77,76.29],[72.56,80.07]]
mnist2_4_acc = 98.51

mnist2_5_it = [[95.81,96.7],[97.34,96.29]]
mnist2_5_mo = [[71.36,72.96],[72.94,75.9]]
mnist2_5_acc = 97.62

mnist2_6_it = [[66.44,65.5],[61.56,59.51]]
mnist2_6_mo = [[73.87,74.24],[74.95,76.1]]
mnist2_6_acc = 97.07

mnist2_7_it = [[],[],[]]
mnist2_7_mo = [[],[],[]]
mnist2_7_acc = 0




c_1_it = [[55.12,48.27],[59.14,54.77]]
c_1_mo = [[69.68,68.1],[68.52,67.56]]
c_1_acc = 79.0

c_2_it = [[73.53,73.45],[72.86,73.67]]
c_2_mo = [[71.27,71.62],[69.81,70.44]]
c_2_acc = 79.60

c_3_it = [[74.87,74.6],[74.66,74.57]]
c_3_mo = [[71.23,71.58],[70.09,71.04]]
c_3_acc = 79.61

c_4_it = [[73.79,73.56],[73.85,73.95]]
c_4_mo = [[66.31,66.95],[65.78,66.4]]
c_4_acc = 75.64

c_5_it = [[67.88,67.57],[68.62,67.98]]
c_5_mo = [[61.62,60.22],[59.53,60.35]]
c_5_acc = 69.33

c_6_it = [[50.96,51.36],[48.56,48.62]]
c_6_mo = [[53.01,52.88],[52.17,52.99]]
c_6_acc = 64.84

c_7_it = [[47.98,59.12,48.9],[63.14,75.88,64.71],[53.29,63.93,50.01]]
c_7_mo = [[77.89,78.03,77.53],[78.10,74.77,77.01],[77.39,74.67,76.09]]
c_7_acc = 79.50


#mnist1
attm1 = [[58.98,73.01],[70.24,54.20]]
feam1 = [[93.16,77.55],[75.09,91.65]]
ascorem1 = [[56.98,66.01],[67.24,54.20]]

#mnist2
attm2 = [[59.90,80.31],[88.65,74.93]]
feam2 = [[92.44,85.36],[77.67,95.04]]
ascorem2 = [[59.90,80.31],[88.65,74.93]]

#cifar
attc = [[63.83,61.53],[63.97,62.10]]
feac = [[71.54,72.83],[68.29,69.23]]
ascorec =[[63.83,61.53],[63.97,62.10]]

accm1 = 98.56
accm2 = 99.08
accc = 79.66


f = calNorm(mo2fea(feac,accc,2))
itx = copy.deepcopy(attc)
att7 = calNorm(attc)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, accc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_1_mo,c_1_acc,2))
itx = copy.deepcopy(c_1_it)
att7 = calNorm(c_1_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_2_mo,c_2_acc,2))
itx = copy.deepcopy(c_2_it)
att7 = calNorm(c_2_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_3_mo,c_3_acc,2))
itx = copy.deepcopy(c_3_it)
att7 = calNorm(c_3_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_4_mo,c_4_acc,2))
itx = copy.deepcopy(c_4_it)
att7 = calNorm(c_4_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_5_mo,c_5_acc,2))
itx = copy.deepcopy(c_5_it)
att7 = calNorm(c_5_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1,c_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_6_mo,c_6_acc,2))
itx = copy.deepcopy(c_6_it)
att7 = calNorm(c_6_it)
score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

# f = calNorm(mo2fea(c_7_mo,c_7_acc,2))
# itx = copy.deepcopy(c_7_it)
# att7 = calNorm(c_7_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, c_7_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")



# print("MNIST1111111111111111111111111")


# f = calNorm(mo2fea(feam1,accm1,2))
# itx = copy.deepcopy(attm1)
# att7 = calNorm(attm1)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, accm1)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,2))
# itx = copy.deepcopy(mnist1_1_it)
# att7 = calNorm(mnist1_1_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_2_mo,mnist1_2_acc,2))
# itx = copy.deepcopy(mnist1_2_it)
# att7 = calNorm(mnist1_2_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_2_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_3_mo,mnist1_3_acc,2))
# itx = copy.deepcopy(mnist1_3_it)
# att7 = calNorm(mnist1_3_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_3_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_4_mo,mnist1_4_acc,2))
# itx = copy.deepcopy(mnist1_4_it)
# att7 = calNorm(mnist1_4_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_4_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_5_mo,mnist1_5_acc,2))
# itx = copy.deepcopy(mnist1_5_it)
# att7 = calNorm(mnist1_5_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_5_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_6_mo,mnist1_6_acc,2))
# itx = copy.deepcopy(mnist1_6_it)
# att7 = calNorm(mnist1_6_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist1_6_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# # f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,3))
# # itx = copy.deepcopy(mnist1_1_it)
# # att7 = calNorm(mnist1_1_it)
# # score7, r7, a7 = calScore(att7,itx, f, 3, -1, 1, mnist1_1_acc)
# # print("!!!!!!!!!!!!!!!!!!!!!!!!")
# # print(score7, r7, a7)
# # print("\r\n")

# print("MNIST222222222222222222222")

# f = calNorm(mo2fea(feam2,accm2,2))
# itx = copy.deepcopy(attm2)
# att7 = calNorm(attm2)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, accm2)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")


# f = calNorm(mo2fea(mnist2_1_mo,mnist2_1_acc,2))
# itx = copy.deepcopy(mnist2_1_it)
# att7 = calNorm(mnist2_1_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_2_mo,mnist2_2_acc,2))
# itx = copy.deepcopy(mnist2_2_it)
# att7 = calNorm(mnist2_2_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_2_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_3_mo,mnist2_3_acc,2))
# itx = copy.deepcopy(mnist2_3_it)
# att7 = calNorm(mnist2_3_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_3_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_4_mo,mnist2_4_acc,2))
# itx = copy.deepcopy(mnist2_4_it)
# att7 = calNorm(mnist2_4_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_4_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_5_mo,mnist2_5_acc,2))
# itx = copy.deepcopy(mnist2_5_it)
# att7 = calNorm(mnist2_5_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_5_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_6_mo,mnist2_6_acc,2))
# itx = copy.deepcopy(mnist2_6_it)
# att7 = calNorm(mnist2_6_it)
# score7, r7, a7 = calScore(att7,itx, f, 2, -1, 1, mnist2_6_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")