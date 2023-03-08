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
mnist1_1_it = [[9.92,17.71,24.23,8.97],[21.15,52.10,61.69,26.82],[32.94,67.39,56.93,15.87],[10.23,21.72,10.38,1.9]]
mnist1_1_mo = [[98.4,97.62,97.93,98.38],[98.06,95.78,91.12,97.97],[98,91.26,96.54,98.09],[98.43,97.72,97.44,98.37]]
mnist1_1_acc = 98.40
#9840

mnist1_2_it = [[11.57,28.63,30.14,22.64],[27.82,71.78,86.96,48.88],[46.47,88.36,80.43,30.06],[15.61,38.55,25.28,5.65]]
mnist1_2_mo = [[97.99,96.89,96.57,97.98],[97.74,94.58,92.13,97.28],[97.57,90.85,94.97,97.69],[97.87,95.8,96.2,97.86]]
mnist1_2_acc = 97.91

mnist1_3_it = [[18.28,47.0,47.97,38.58],[53.86,89.97,91.72,67.91],[65.79,93.41,89.93,59.27],[34.23,47.09,40.93,10.59]]
mnist1_3_mo = [[97.65,96.54,96.93,97.70],[97.28,91.47,90.85,96.36],[97.09,85.65,93.3,97.37],[97.87,95.7,93.53,97.39]]
mnist1_3_acc = 97.65

mnist1_4_it = [[77.28,85.89,86.57,77.26],[90.1,94.52,94.55,90.56],[90.28,95.29,94.54,90.05],[80.14,88.87,88.61,79.92]]
mnist1_4_mo = [[94.92,92.07,91.08,95.04],[93.71,84.13,82.55,93.68],[93.53,83.51,84.11,94.12],[95.16,92.39,92.18,95.06]]
mnist1_4_acc = 95.26

mnist1_5_it = [[87.91,91.33,91.35,88.94],[91.56,94.22,94.61,91.98],[91.9,94.78,94.39,92.68],[88.51,91.06,90.58,89.33]]
mnist1_5_mo = [[92.51,88.37,90.12,92.82],[91.07,79.50,80.62,91.19],[90.76,78.96,78.61,90.98],[91.93,88.15,88.4,92.48]]
mnist1_5_acc = 92.99


mnist1_6_it = [[73.93,77.35,79.63,80.34],[79.73,79.82,79.94,81.65],[76.59,75.75,75.83,78.09],[78.06,79.14,79.07,78.87]]
mnist1_6_mo = [[89.06,82.96,84.7,87.66],[85.5,71.37,74.59,85.36],[84.48,74.37,75.26,85.72],[89.25,86.68,86.72,88.48]]
mnist1_6_acc = 89.76

mnist1_7_it = [[],[],[]]
mnist1_7_mo = [[],[],[]]
mnist1_7_acc = 0




mnist2_1_it = [[10.9,19.01,30.23,18.84],[24.46,53.01,63.73,40.38],[42.66,86.63,68.42,12.08],[10.79,21.19,21.14,2.25]]
mnist2_1_mo = [[99.12,98.9,98.89,99.08],[99.15,96.39,87.5,98.35],[99.1,85.52,94.7,98.67],[99.18,96.3,98.43,99.02]]
mnist2_1_acc = 99.12

mnist2_2_it = [[14.45,36.82,38.93,27.59],[51.31,92.23,94.06,65.23],[58.72,95.87,94.4,50.8],[18.77,33.62,29.61,8.76]]
mnist2_2_mo = [[99.02,98.86,98.46,98.99],[98.9,90.95,92.81,97.82],[98.84,81.75,94.36,98.71],[99.08,96.29,97.21,98.88]]
mnist2_2_acc = 99.05

mnist2_3_it = [[17.4,49.23,54.65,35.66],[72.78,96.7,96.59,80.8],[73.75,97.53,96.79,74.46],[31.98,54.74,56.16,16.8]]
mnist2_3_mo = [[99.04,98.62,97.84,98.83],[98.9,88.89,90.43,97.68],[98.83,85.07,93.65,98.59],[99.11,97.75,97.31,98.87]]
mnist2_3_acc = 98.90

mnist2_4_it = [[89.77,94.90,94.52,90.62],[96.74,98.24,98.23,96.67],[96.87,98.28,98.20,96.52],[93.38,95.57,95.35,90.91]]
mnist2_4_mo = [[98.52,97.50,97.22,97.99],[97.8,86.36,87.33,96.3],[97.78,79.48,87.21,96.77],[98.23,93.3,94.35,98.02]]
mnist2_4_acc = 98.16

mnist2_5_it = [[96.47,97.29,97.23,96.75],[97.3,98.11,98.3,97.74],[97.53,98.3,98.26,97.5],[96.62,97.04,96.88,96.04]]
mnist2_5_mo = [[97.74,96.41,96.2,96.98],[96.88,87.47,85.02,93.53],[96.53,82.78,83.57,94],[97.62,92.95,92.59,96.94]]
mnist2_5_acc = 97.62

mnist2_6_it = [[91.31,93.32,92.72,92.29],[91.55,93.4,92.81,92.74],[91.97,93.2,92.78,92.83],[91.71,92.57,91.82,91.91]]
mnist2_6_mo = [[96.67,95.25,95.24,96.24],[95.95,86.57,84.95,92.27],[95.54,83.68,82.62,92.14],[96.43,91.66,90.92,95.35]]
mnist2_6_acc = 97.00

mnist2_7_it = [[],[],[]]
mnist2_7_mo = [[],[],[]]
mnist2_7_acc = 0




c_1_it = [[],[],[],[]]
c_1_mo = [[],[],[],[]]
c_1_acc = 79.19

c_2_it = [[],[],[],[]]
c_2_mo = [[],[],[],[]]
c_2_acc = 79.87

c_3_it = [[],[],[],[]]
c_3_mo = [[],[],[],[]]
c_3_acc = 79.32

c_4_it = [[],[],[],[]]
c_4_mo = [[],[],[],[]]
c_4_acc = 75.35

c_5_it = [[67.05,69.3,69.5,67.32],[70.62,72.32,72.52,70.29],[70.58,72.41,72.41,70.14],[67.48,69.35,68.8,66.93]]
c_5_mo = [[70.81,69.96,69.92,70.5],[70.71,66.3,65.79,68.96],[69.31,65.21,64.74,69.32],[70.68,68.23,68.26,70.03]]
c_5_acc = 71.63

c_6_it = [[],[],[],[]]
c_6_mo = [[],[],[],[]]
c_6_acc = 65.75

c_7_it = [[47.98,59.12,48.9],[63.14,75.88,64.71],[53.29,63.93,50.01]]
c_7_mo = [[77.89,78.03,77.53],[78.10,74.77,77.01],[77.39,74.67,76.09]]
c_7_acc = 79.50


#mnist1
attm1 = [[10,17.89,23.5,11.09],[17.98,48.33,54.86,19.44],[29.36,55.94,18.73,19.66],[15.09,15.18,17.01,4.983]]
feam1 = [[98.41,96.18,94.30,98.4],[98.32,89.26,86.87,97.92],[98.05,88.74,95.67,98.31],[98.39,97.36,98.10,98.43]]
ascorem1 = [[10,17.89,23.5,11.09],[17.98,48.33,54.86,19.44],[29.36,55.94,18.73,19.66],[15.09,15.18,17.01,4.983]]

#mnist2
attm2 = [[11.55,15.62,24.93,13.13],[27.77,51.96,56.71,32.09],[35.60,76.23,60.17,16.39],[8.031,13.63,14.9,2.997]]
feam2 = [[99.18,98.98,98.87,99.13],[99.19,94.69,83.92,98.06],[99.11,82.32,95.81,98.77],[99.21,94.01,98.30,99.01]]
ascorem2 = [[11.55,15.62,24.93,13.13],[27.77,51.96,56.71,32.09],[35.60,76.23,60.17,16.39],[8.031,13.63,14.9,2.997]]

#cifar
attc = [[28.94,36.62,36.80,30.92],[36.8,46.37,45.72,38.40],[38.9,48.73,49.99,36.96],[35.83,48.33,44.86,34.29]]
feac = [[78.39,77.95,78.11,78.2],[77.89,76.84,76.27,78.21],[78.11,76.25,77.05,77.80],[78.23,77.25,77.29,78.07]]
ascorec = [[28.94,36.62,36.80,30.92],[36.8,46.37,45.72,38.40],[38.9,48.73,49.99,36.96],[35.83,48.33,44.86,34.29]]

accm1 = 98.56
accm2 = 99.08
accc = 79.66


f = calNorm(mo2fea(feac,accc,4))
itx = copy.deepcopy(attc)
att7 = calNorm(attc)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, accc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_1_mo,c_1_acc,4))
itx = copy.deepcopy(c_1_it)
att7 = calNorm(c_1_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_2_mo,c_2_acc,4))
itx = copy.deepcopy(c_2_it)
att7 = calNorm(c_2_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_3_mo,c_3_acc,4))
itx = copy.deepcopy(c_3_it)
att7 = calNorm(c_3_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_4_mo,c_4_acc,4))
itx = copy.deepcopy(c_4_it)
att7 = calNorm(c_4_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_5_mo,c_5_acc,4))
itx = copy.deepcopy(c_5_it)
att7 = calNorm(c_5_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1,c_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_6_mo,c_6_acc,4))
itx = copy.deepcopy(c_6_it)
att7 = calNorm(c_6_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

# f = calNorm(mo2fea(c_7_mo,c_7_acc,4))
# itx = copy.deepcopy(c_7_it)
# att7 = calNorm(c_7_it)
# score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, c_7_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")



print("MNIST1111111111111111111111111")


f = calNorm(mo2fea(feam1,accm1,4))
itx = copy.deepcopy(attm1)
att7 = calNorm(attm1)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, accm1)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,4))
itx = copy.deepcopy(mnist1_1_it)
att7 = calNorm(mnist1_1_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_2_mo,mnist1_2_acc,4))
itx = copy.deepcopy(mnist1_2_it)
att7 = calNorm(mnist1_2_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_3_mo,mnist1_3_acc,4))
itx = copy.deepcopy(mnist1_3_it)
att7 = calNorm(mnist1_3_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_4_mo,mnist1_4_acc,4))
itx = copy.deepcopy(mnist1_4_it)
att7 = calNorm(mnist1_4_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_5_mo,mnist1_5_acc,4))
itx = copy.deepcopy(mnist1_5_it)
att7 = calNorm(mnist1_5_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_6_mo,mnist1_6_acc,4))
itx = copy.deepcopy(mnist1_6_it)
att7 = calNorm(mnist1_6_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

# f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,4))
# itx = copy.deepcopy(mnist1_1_it)
# att7 = calNorm(mnist1_1_it)
# score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist1_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

print("MNIST222222222222222222222")

f = calNorm(mo2fea(feam2,accm2,4))
itx = copy.deepcopy(attm2)
att7 = calNorm(attm2)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, accm2)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")


f = calNorm(mo2fea(mnist2_1_mo,mnist2_1_acc,4))
itx = copy.deepcopy(mnist2_1_it)
att7 = calNorm(mnist2_1_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_2_mo,mnist2_2_acc,4))
itx = copy.deepcopy(mnist2_2_it)
att7 = calNorm(mnist2_2_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_3_mo,mnist2_3_acc,4))
itx = copy.deepcopy(mnist2_3_it)
att7 = calNorm(mnist2_3_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_4_mo,mnist2_4_acc,4))
itx = copy.deepcopy(mnist2_4_it)
att7 = calNorm(mnist2_4_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_5_mo,mnist2_5_acc,4))
itx = copy.deepcopy(mnist2_5_it)
att7 = calNorm(mnist2_5_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_6_mo,mnist2_6_acc,4))
itx = copy.deepcopy(mnist2_6_it)
att7 = calNorm(mnist2_6_it)
score7, r7, a7 = calScore(att7,itx, f, 4, -1, 1, mnist2_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")