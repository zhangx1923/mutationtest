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
    for i in range(n):
        for j in range(n):
            sum_f += (f[i][j] - avg) * (f[i][j] - avg)
            sum_a += (a[i][j] - avg) * (a[i][j] - avg)
            sum_af += (f[i][j] - a[i][j]) * (f[i][j] - a[i][j])
            sum_aaa += (ac/100-abefore[i][j]/100) * (ac/100-abefore[i][j]/100)
    robust = math.sqrt(sum_f)/9 + math.sqrt(sum_a)/9+ math.sqrt(sum_aaa)/9
    # print(math.sqrt(sum_f), math.sqrt(sum_a))
    #robust = sigmoid(robust)
    acc = ac/100-(math.sqrt(sum_af)/9 )
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
attm1 = [[0.0638,0.1078,0.0550],[0.1333,0.2782,0.1284],[0.0841,0.1169,0.0325]]
feam1 = [[0.0105,0.2299,0.0231],[0.091,0.4329,0.0386],[0.02,0.1394,0.0145]]
ascorem1 = [[21.65,36.56,18.66],[45.23,94.36,43.54],[28.51,39.65,11.03]]

#mnist2
attm2 = [[0.0539,0.1073,0.076],[0.1441,0.2471,0.161],[0.0769,0.1081,0.0257]]
feam2 = [[0.0017,0.086,0.0122],[0.029,0.491,0.0915],[0.0037,0.2729,0.012]]
ascorem2 = [[21.03,41.85,29.63],[56.20,96.39,62.8],[30.01,42.18,10.03]]

#cifar
attc = [[0.0959,0.1102,0.0927],[0.1161,0.1415,0.1154],[0.1004,0.125,0.1027]]
feac = [[0.0709,0.112,0.0587],[0.1096,0.2647,0.1078],[0.0792,0.1259,0.0633]]
ascorec = [[42.31,48.61,40.9],[51.22,62.4,50.88],[44.29,55.11,45.28]]

accm1 = 98.56
loss1 = 0.0413

accm2 = 99.08
loss2 = 0.0149

accc = 79.66
loss3 = 0.7049

score7, r7, a7 = calScore1(attc,ascorec, feac, 3, -1, 1, accc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_1_mo,c_1_acc,3))
itx = copy.deepcopy(c_1_it)
att7 = calNorm(c_1_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_2_mo,c_2_acc,3))
itx = copy.deepcopy(c_2_it)
att7 = calNorm(c_2_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_3_mo,c_3_acc,3))
itx = copy.deepcopy(c_3_it)
att7 = calNorm(c_3_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_4_mo,c_4_acc,3))
itx = copy.deepcopy(c_4_it)
att7 = calNorm(c_4_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_5_mo,c_5_acc,3))
itx = copy.deepcopy(c_5_it)
att7 = calNorm(c_5_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1,c_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(c_6_mo,c_6_acc,3))
itx = copy.deepcopy(c_6_it)
att7 = calNorm(c_6_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, c_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

# f = calNorm(mo2fea(c_7_mo,c_7_acc,3))
# itx = copy.deepcopy(c_7_it)
# att7 = calNorm(c_7_it)
# score7, r7, a7 = calScore(att7,itx, f, 3, -1, 1, c_7_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")



print("MNIST1111111111111111111111111")



score7, r7, a7 = calScore1(attm1,ascorem1, feam1, 3, -1, 1, accm1)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,3))
itx = copy.deepcopy(mnist1_1_it)
att7 = calNorm(mnist1_1_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_2_mo,mnist1_2_acc,3))
itx = copy.deepcopy(mnist1_2_it)
att7 = calNorm(mnist1_2_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_3_mo,mnist1_3_acc,3))
itx = copy.deepcopy(mnist1_3_it)
att7 = calNorm(mnist1_3_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_4_mo,mnist1_4_acc,3))
itx = copy.deepcopy(mnist1_4_it)
att7 = calNorm(mnist1_4_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_5_mo,mnist1_5_acc,3))
itx = copy.deepcopy(mnist1_5_it)
att7 = calNorm(mnist1_5_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist1_6_mo,mnist1_6_acc,3))
itx = copy.deepcopy(mnist1_6_it)
att7 = calNorm(mnist1_6_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist1_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

# f = calNorm(mo2fea(mnist1_1_mo,mnist1_1_acc,3))
# itx = copy.deepcopy(mnist1_1_it)
# att7 = calNorm(mnist1_1_it)
# score7, r7, a7 = calScore(att7,itx, f, 3, -1, 1, mnist1_1_acc)
# print("!!!!!!!!!!!!!!!!!!!!!!!!")
# print(score7, r7, a7)
# print("\r\n")

print("MNIST222222222222222222222")

score7, r7, a7 = calScore1(attm2,ascorem2, feam2, 3, -1, 1, accm2)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")


f = calNorm(mo2fea(mnist2_1_mo,mnist2_1_acc,3))
itx = copy.deepcopy(mnist2_1_it)
att7 = calNorm(mnist2_1_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_1_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_2_mo,mnist2_2_acc,3))
itx = copy.deepcopy(mnist2_2_it)
att7 = calNorm(mnist2_2_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_2_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_3_mo,mnist2_3_acc,3))
itx = copy.deepcopy(mnist2_3_it)
att7 = calNorm(mnist2_3_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_3_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_4_mo,mnist2_4_acc,3))
itx = copy.deepcopy(mnist2_4_it)
att7 = calNorm(mnist2_4_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_4_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_5_mo,mnist2_5_acc,3))
itx = copy.deepcopy(mnist2_5_it)
att7 = calNorm(mnist2_5_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_5_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")

f = calNorm(mo2fea(mnist2_6_mo,mnist2_6_acc,3))
itx = copy.deepcopy(mnist2_6_it)
att7 = calNorm(mnist2_6_it)
score7, r7, a7 = calScore1(att7,itx, f, 3, -1, 1, mnist2_6_acc)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
print(score7, r7, a7)
print("\r\n")