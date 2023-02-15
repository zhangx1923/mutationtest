import math

#it1-6   cifar的六种增强方法
it1 = [[45.32,52.1,45.11],[56.63,66.98,57.91],[49.1,58.23,50.17]]
mo1 = [[80.1,77.93,79.32],[79.12,73.2,78.61],[78.1,74.77,78.22]]
ac1 = 81.22

it2 = [[40.99,47.32,36.68],[50.43,58.73,45.98],[44.33,49.17,40.5]]
mo2 = [[72.91,70.01,72.4],[70.56,64.82,71.0],[71.17,68.33,72.6]]
ac2 = 73.7

it3 = [[32.1,38.17,27.98],[41.23,45.96,34.47],[35.6,37.88,26.35]]
mo3 = [[64.97,64.2,64.33],[62.77,58.17,60.9],[62.1,62.61,64.33]]
ac3 = 67.1

it4 = [[38.01,45.11,39.81],[47.3,57.63,47.83],[43.26,49.71,39.99]]
mo4 = [[73.8,71.0,72.31],[72.79,67.66,71.58],[71.13,69.84,72.09]]
ac4 = 75.66

it5 = [[69.64,71.97,70.07],[72.8,75.09,72.59],[69.26,72.04,69.56]]
mo5 = [[73.09,70.05,72.44],[71.85,65.54,70.65],[71.96,69.25,71.57]]
ac5 = 74.11

it6 = [[68.01,71.23,70.65],[71.1,73.03,72.32],[70.0,71.16,70.99]]
mo6 = [[71.54,68.09,69.95],[68.43,64.81,67.28],[69.77,65.55,68.63]]
ac6 = 71.04

# it = [[],[],[]]
# mo = [[],[],[]]
# ac = 

#mnist1
attm1 = [[0.0638,0.1078,0.550],[0.1333,0.2782,0.1284],[0.0841,0.1169,0.0325]]
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


#it7,8 mnist1的两种增强
it7 = [[90,93,90],[93,96,93],[92,93,91]]
mo7 = [[93,81,93],[88,68,87],[93,83,93]]
ac7 = 94.64

it8 = [[91,91,89],[91,92,90],[91,92,91]]
mo8 = [[89,79,89],[83,53,84],[90,74,89]]
ac8 = 92.94

#it 9,10 mnist2的两种增强  9 p=0.5  10 p=1
it9 = [[84,96,82],[98,99,98],[84,95,79]]
mo9 = [[98,91,98],[95,60,92],[97,85,97]]
ac9 = 98.59

it10 = [[97,97,97],[97,97,97],[97,97,97]]
mo10 = [[98,91,96],[95,71,90],[96,83,95]]
ac10 = 96.58







accm1 = 98.56
loss1 = 0.0413

accm2 = 99.08
loss2 = 0.0149

accc = 79.66
loss3 = 0.7049

def mo2fea(a, ac, n):
    for i in range(n):
        for j in range(n):
            a[i][j] = ac-a[i][j] if ac-a[i][j] > 0 else 0
    return a

import numpy as np
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

import copy
alpha = -1
beta = 1
n = 3

x,y,z = calScore(attm1,ascorem1, feam1, n, alpha, beta, accm1)
x2,y2,z2 = calScore(attm2,ascorem2, feam2, n, alpha, beta, accm2)
x3,y3,z3 = calScore(attc,ascorec, feac, n, alpha, beta, accc)

print(x,y,z)
fea7 = calNorm(mo2fea(mo7,ac7,n))
itx = copy.deepcopy(it7)
att7 = calNorm(it7)
score7, r7, a7 = calScore(att7,itx, fea7, n, alpha, beta, ac7)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea1, att1)
print(score7, r7, a7)
print("\r\n")
#print(sigmoid(score1))

fea8 = calNorm(mo2fea(mo8,ac8,n))
itx = copy.deepcopy(it8)
att8 = calNorm(it8)
score8, r8, a8 = calScore(att8,itx, fea8, n, alpha, beta, ac8)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea1, att1)
print(score8, r8, a8)
print("\r\n")
#print(sigmoid(score1))

print(x2,y2,z2)

fea9 = calNorm(mo2fea(mo9,ac9,n))
itx = copy.deepcopy(it9)
att9 = calNorm(it9)
score9, r9, a9 = calScore(att9,itx, fea9, n, alpha, beta, ac9)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea1, att1)
print(score9, r9, a9)
print("\r\n")
#print(sigmoid(score1))

fea10 = calNorm(mo2fea(mo10,ac10,n))
itx = copy.deepcopy(it10)
att10 = calNorm(it10)
score10, r10, a10 = calScore(att10,itx, fea10, n, alpha, beta, ac10)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea1, att1)
print(score10, r10, a10)
print("\r\n")
#print(sigmoid(score1))

print("finist mnist")




print(x3,y3,z3)


fea1 = calNorm(mo2fea(mo1,ac1,n))
itx = copy.deepcopy(it1)
att1 = calNorm(it1)

score1, r1, a1 = calScore(att1,itx, fea1, n, alpha, beta, ac1)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea1, att1)
print(score1, r1, a1)
print("\r\n")
#print(sigmoid(score1))


fea2 = calNorm(mo2fea(mo2,ac2,n))
itx = copy.deepcopy(it2)
att2 = calNorm(it2)
score2, r2, a2 = calScore(att2,itx, fea2, n, alpha, beta, ac2)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea2, att2)
print(score2, r2, a2)
print("\r\n")
#print(sigmoid(score2))

fea3 = calNorm(mo2fea(mo3,ac3,n))
itx = copy.deepcopy(it3)
att3 = calNorm(it3)
score3, r3, a3 = calScore(att3,itx, fea3, n, alpha, beta, ac3)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea3, att3)
print(score3, r3, a3)
print("\r\n")
#print(sigmoid(score3))

fea4 = calNorm(mo2fea(mo4,ac4,n))
itx = copy.deepcopy(it4)
att4 = calNorm(it4)
score4, r4, a4 = calScore(att4,itx, fea4, n, alpha, beta, ac4)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea4, att4)
print(score4, r4, a4)
print("\r\n")
#print(sigmoid(score4))

fea5 = calNorm(mo2fea(mo5,ac5,n))
itx = copy.deepcopy(it5)
att5 = calNorm(it5)
score5, r5, a5 = calScore(att5,itx, fea5, n, alpha, beta, ac5)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea5, att5)
print(score5, r5, a5)
print("\r\n")
#print(sigmoid(score5))

fea6 = calNorm(mo2fea(mo6,ac6,n))
itx = copy.deepcopy(it6)
att6 = calNorm(it6)
score6, r6, a6 = calScore(att6,itx, fea6, n, alpha, beta, ac6)
print("!!!!!!!!!!!!!!!!!!!!!!!!")
#print(fea6, att6)
print(score6, r6, a6)
print("\r\n")
##print(sigmoid(score6))
