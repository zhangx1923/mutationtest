import numpy as np
import matplotlib.pyplot as plt
from sympy import C


o_cm = [0.6278,0.6709,0.6962, 0.6974,0.6943,0.6473,0.5847]
r_cm = [0.1546,0.1165,0.0869, 0.0865,0.0362,0.0272,0.0512]
a_cm = [0.7825,0.7874,0.7831, 0.7839,0.7305,0.6745,0.6359]

o_m1 = [0.6731,0.7629,0.8088,0.8643,0.8682 ,0.8463,0.7627]
r_m1 = [0.2768,0.1833,0.1267,0.0509,0.0497 ,0.0446,0.1085]
a_m1 = [0.9499,0.9462,0.9356,0.9152,0.9179 ,0.8910,0.8706]

o_m2 = [0.6876,0.7958,0.8477,0.8966,0.9028 ,0.9012,0.8497]
r_m2 = [0.2649,0.1474,0.0974,0.0452,0.0418 ,0.0387,0.0909]
a_m2 = [0.9526,0.9433,0.9451,0.9418,0.9416 ,0.9400,0.9407]


labels = ['p=0.1','p=0.2','p=0.25','p=0.26','p=0.5','p=0.75','p=1.0']
labelsm1 = ['p=0.1','p=0.2','p=0.25','p=0.5','p=0.56','p=0.75','p=1.0']
labelsm2 = ['p=0.1','p=0.2','p=0.25','p=0.5','p=0.55','p=0.75','p=1.0']
overall = o_m1
robust = r_m1
acc = a_m1

plt.figure(figsize=(20,5))
#plot 1:

ax1 = plt.subplot(1, 3, 1)
plt.plot(labels,o_cm,color="red",label="D-Score")
plt.plot(labels,a_cm,color="green",label="acc")
plt.plot(labels,r_cm,color="blue", label="robust")
plt.title("CM")
plt.legend(fontsize=14)
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

#plot 2:

plt.subplot(1, 3, 2)
plt.plot(labelsm1,o_m1,color="red",label="D-Score")
plt.plot(labelsm1,a_m1,color="green",label="acc")
plt.plot(labelsm1,r_m1,color="blue", label="robust")
plt.title("MMA")
plt.legend(fontsize=14,loc='center left')
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
#plot 2:

plt.subplot(1, 3, 3)
plt.plot(labelsm2,o_m2,color="red",label="D-Score")
plt.plot(labelsm2,a_m2,color="green",label="acc")
plt.plot(labelsm2,r_m2,color="blue", label="robust")
plt.title("MMB")
plt.legend(fontsize=14,loc='center left')
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()


plt.savefig("bar.png")





# barWidth = 0.75
# r1 = np.arange(len(overall))
# r2 = [x+ barWidth for x in r1]
# r3 = [x + barWidth for x in r2]

# def autolabel(rects):
#     for rect in rects:
#         height = rect.get_height()
#         plt.text(rect.get_x()-0.08, 1.02*height, "%s" % height, size=10)

# plt.figure(figsize=(6,4))
# cm = plt.bar(r1,overall,color="red",width=barWidth,edgecolor='white',label='overall')
# autolabel(cm)
# cm = plt.bar(r2,robust,color="blue",width=barWidth,edgecolor='white',label='robust')
# autolabel(cm)
# cm = plt.bar(r3,acc,color="green",width=barWidth,edgecolor='white',label='acc')
# autolabel(cm)

# plt.ylim((0,1))
# plt.xlabel("score type")
# plt.ylabel("probability")
# plt.yticks(size=10)
# plt.xticks([r+barWidth for r in range(len(overall))],labels=labels, size=10)

#plt.legend()
# plt.savefig("bar.jpg")