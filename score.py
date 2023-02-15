import math

att1 = [[0.0638,0.1078,0.550],[0.1333,0.2782,0.1284],[0.0841,0.1169,0.0325]]
fea1 = [[0.0105,0.2299,0.0231],[0.091,0.4329,0.0386],[0.02,0.1394,0.0145]]

att2 = [[0.0539,0.1073,0.076],[0.1441,0.2471,0.161],[0.0769,0.1081,0.0257]]
fea2 = [[0.0017,0.086,0.0122],[0.029,0.491,0.0915],[0.0037,0.2729,0.012]]

att3 = [[0.0959,0.1102,0.0927],[0.1161,0.1415,0.1154],[0.1004,0.125,0.1027]]
fea3 = [[0.0709,0.112,0.0587],[0.1096,0.2647,0.1078],[0.0792,0.1259,0.0633]]

acc1 = 0.9856
loss1 = 0.0413

acc2 = 0.9908
loss2 = 0.0149

acc3 = 0.7966
loss3 = 0.7049

import numpy as np
def sigmoid(x):
    return 1.0/(1+np.exp(-x))

def std(a,n):
    arr = []
    for i in range(n):
        for j in range(n):
            arr.append(a[i][j])
    return np.std(arr)

def calScore2(a, f, n, alpha, beta, ac, loss):
    s = 0
    sum_f = 0
    sum_a = 0
    sum_af = 0
    avg = 1/(n*n)
    for i in range(n):
        for j in range(n):
            sum_f += (f[i][j] - avg) * (f[i][j] - avg)
            sum_a += (a[i][j] - avg) * (a[i][j] - avg)
            sum_af += (f[i][j] - a[i][j]) * (f[i][j] - a[i][j])

    robust = math.sqrt(sum_f) + math.sqrt(sum_a)
    robust = sigmoid(robust)
    acc = math.sqrt(sum_af)
    acc = sigmoid(acc)

    # acc_para = ac/loss
    # acc_para = sigmoid(acc_para)
    acc_para = acc-sigmoid(loss)

    acc_final = (acc+acc_para)

    s = alpha*robust + beta*acc_final
    return s, robust, acc_final, acc, std(a,n), std(f,n)

def calScore(a, f, n, alpha, beta, ac):
    s = 0
    sum_f = 0
    sum_a = 0
    sum_af = 0
    avg = 1/(n*n)
    for i in range(n):
        for j in range(n):
            sum_f += (f[i][j] - avg) * (f[i][j] - avg)
            sum_a += (a[i][j] - avg) * (a[i][j] - avg)
            sum_af += (f[i][j] - a[i][j]) * (f[i][j] - a[i][j])
    robust = math.sqrt(sum_f) + math.sqrt(sum_a)
    #robust = sigmoid(robust)
    acc = ac - math.sqrt(sum_af)/5
    #acc = sigmoid(acc)
    s = alpha*robust + beta*acc
    return s, robust, acc

def calScore1(a, f, n, alpha, beta):
    s = 0
    sum_f = 0
    sum_a = 0
    sum_af = 0
    avg = 1/(n*n)
    for i in range(n):
        for j in range(n):
            sum_f += abs(f[i][j] - avg)
            sum_a += abs(a[i][j] - avg)
            sum_af += abs(f[i][j] - a[i][j]) 
    robust = sum_f + sum_a
    acc = sum_af
    s = alpha*robust + beta*acc
    return s, robust, acc 

alpha = -1
beta = 1
n = 3
score1, r1, a1 = calScore(att1, fea1, n, alpha, beta, acc1)
score2, r2, a2 = calScore(att2, fea2, n, alpha, beta, acc2)
score3, r3, a3 = calScore(att3, fea3, n, alpha, beta, acc3)

# score1, r1, af1, a1, std1, std11 = calScore2(att1, fea1, n, alpha, beta, acc1, loss1)
# score2, r2, af2, a2, std2, std22 = calScore2(att2, fea2, n, alpha, beta, acc2, loss2)
# score3, r3, af3, a3, std3, std33 = calScore2(att3, fea3, n, alpha, beta, acc3, loss3)

print(score1, r1, a1)
print("\r\n")
print(score2, r2, a2)
print("\r\n")
print(score3, r3, a3)



print(sigmoid(score1), sigmoid(score2), sigmoid(score3))