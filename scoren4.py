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
mnist1_1_it = [[20,44,32],[47,92,51],[27,43,11]]
mnist1_1_mo = [[98,91,98],[94,68,94],[98,87,98]]
mnist1_1_acc = 98.40
#9840

mnist1_2_it = [[43,68,47],[84,97,83],[49,69,37]]
mnist1_2_mo = [[98,81,95],[88,71,95],[97,90,97]]
mnist1_2_acc = 97.94

mnist1_3_it = [[60,83,70],[89,97,90],[71,85,61]]
mnist1_3_mo = [[97,86,96],[93,70,93],[97,87,97]]
mnist1_3_acc = 97.59

mnist1_4_it = [[92,94,92],[95,97,94],[93,93,91]]
mnist1_4_mo = [[94,86,94],[91,65,88],[94,71,94]]
mnist1_4_acc = 95.59

mnist1_5_it = [[85,89,88],[91,95,91],[90,91,89]]
mnist1_5_mo = [[89,79,90],[81,57,80],[89,71,88]]
mnist1_5_acc = 92.22


mnist1_6_it = [[65,69,68],[65,68,68],[65,67,69]]
mnist1_6_mo = [[85,73,85],[78,56,79],[87,81,87]]
mnist1_6_acc = 90.12

mnist1_7_it = [[],[],[]]
mnist1_7_mo = [[],[],[]]
mnist1_7_acc = 0




mnist2_1_it = [[18,47,34],[64,97,69],[36,51,9]]
mnist2_1_mo = [[99,93,98],[97,70,94],[99,88,98]]
mnist2_1_acc = 99.12

mnist2_2_it = [[45,76,63],[91,99,92],[64,83,98]]
mnist2_2_mo = [[99,93,98],[96,72,95],[98,90,98]]
mnist2_2_acc = 99.05

mnist2_3_it = [[72,90,80],[96,99,95],[84,94,81]]
mnist2_3_mo = [[99,92,98],[95,68,92],[98,87,97]]
mnist2_3_acc = 98.98

mnist2_4_it = [[97,98,97],[98,99,98],[98,98,97]]
mnist2_4_mo = [[98,90,97],[93,64,87],[97,82,96]]
mnist2_4_acc = 98.38

mnist2_5_it = [[96,97,96],[97,99,98],[97,97,96]]
mnist2_5_mo = [[97,89,94],[92,67,84],[95,82,93]]
mnist2_5_acc = 97.47

mnist2_6_it = [[80,82,82],[74,75,76],[78,78,80]]
mnist2_6_mo = [[95,89,92],[91,72,81],[94,84,89]]
mnist2_6_acc = 96.86

mnist2_7_it = [[],[],[]]
mnist2_7_mo = [[],[],[]]
mnist2_7_acc = 0




c_1_it = [[30,37,26],[42,53,40],[40,52,44]]
c_1_mo = [[77,75,77],[76,72,76],[76,74,76]]
c_1_acc = 79.19

c_2_it = [[39,51,38],[59,74,61],[50,61,52]]
c_2_mo = [[77,76,77],[77,72,77],[77,74,76]]
c_2_acc = 79.87

c_3_it = [[49,61,50],[66,76,64],[57,66,56]]
c_3_mo = [[77,75,77],[76,72,76],[76,74,75]]
c_3_acc = 79.32

c_4_it = [[71,73,71],[74,75,73],[70,73,70]]
c_4_mo = [[74,71,73],[73,66,72],[73,69,73]]
c_4_acc = 75.35

c_5_it = [[69,69,69],[71,72,72],[69,70,69]]
c_5_mo = [[68,66,67],[66,60,65],[68,65,67]]
c_5_acc = 69.62

c_6_it = [[59,57,59],[58,56,58],[57,54,56]]
c_6_mo = [[64,61,63],[61,54,60],[63,59,62]]
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

# f = calNorm(mo2fea(c_1_mo,c_1_acc,3))
# itx = copy.deepcopy(c_1_it)
# att7 = calNorm(c_1_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_2_mo,c_2_acc,3))
# itx = copy.deepcopy(c_2_it)
# att7 = calNorm(c_2_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_2_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_3_mo,c_3_acc,3))
# itx = copy.deepcopy(c_3_it)
# att7 = calNorm(c_3_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_3_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_4_mo,c_4_acc,3))
# itx = copy.deepcopy(c_4_it)
# att7 = calNorm(c_4_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_4_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_5_mo,c_5_acc,3))
# itx = copy.deepcopy(c_5_it)
# att7 = calNorm(c_5_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1,c_5_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_6_mo,c_6_acc,3))
# itx = copy.deepcopy(c_6_it)
# att7 = calNorm(c_6_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_6_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(c_7_mo,c_7_acc,3))
# itx = copy.deepcopy(c_7_it)
# att7 = calNorm(c_7_it)
# score7, r7, a7 = calScore(att7,itx, f, 3, -1, 1, c_7_acc)
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

# f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,3))
# itx = copy.deepcopy(mnist1_1_it)
# att7 = calNorm(mnist1_1_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_2_mo,mnist1_2_acc,3))
# itx = copy.deepcopy(mnist1_2_it)
# att7 = calNorm(mnist1_2_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_2_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_3_mo,mnist1_3_acc,3))
# itx = copy.deepcopy(mnist1_3_it)
# att7 = calNorm(mnist1_3_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_3_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_4_mo,mnist1_4_acc,3))
# itx = copy.deepcopy(mnist1_4_it)
# att7 = calNorm(mnist1_4_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_4_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_5_mo,mnist1_5_acc,3))
# itx = copy.deepcopy(mnist1_5_it)
# att7 = calNorm(mnist1_5_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_5_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_6_mo,mnist1_6_acc,3))
# itx = copy.deepcopy(mnist1_6_it)
# att7 = calNorm(mnist1_6_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_6_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,3))
# itx = copy.deepcopy(mnist1_1_it)
# att7 = calNorm(mnist1_1_it)
# score7, r7, a7 = calScore(att7,itx, f, 3, -1, 1, mnist1_1_acc)
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


# f = calNorm(mo2fea(mnist2_1_mo,mnist2_1_acc,3))
# itx = copy.deepcopy(mnist2_1_it)
# att7 = calNorm(mnist2_1_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_2_mo,mnist2_2_acc,3))
# itx = copy.deepcopy(mnist2_2_it)
# att7 = calNorm(mnist2_2_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_2_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_3_mo,mnist2_3_acc,3))
# itx = copy.deepcopy(mnist2_3_it)
# att7 = calNorm(mnist2_3_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_3_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_4_mo,mnist2_4_acc,3))
# itx = copy.deepcopy(mnist2_4_it)
# att7 = calNorm(mnist2_4_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_4_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_5_mo,mnist2_5_acc,3))
# itx = copy.deepcopy(mnist2_5_it)
# att7 = calNorm(mnist2_5_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_5_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

# f = calNorm(mo2fea(mnist2_6_mo,mnist2_6_acc,3))
# itx = copy.deepcopy(mnist2_6_it)
# att7 = calNorm(mnist2_6_it)
# score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_6_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")