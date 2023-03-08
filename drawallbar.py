import numpy as np
import matplotlib.pyplot as plt
from sympy import C

#n=2
o_cm2 = [0.6622,0.7682,0.7552,0.7358,0.6649,0.5605]
r_cm2 = [0.1108,0.0484,0.0328,0.0151,0.0181,0.0819]
a_cm2 = [0.7730,0.8166,0.7880,0.7509,0.6830,0.6424]

o_m12 = [0.6650,0.8982,0.9103,0.9224,0.8716,0.6578]
r_m12 = [0.2631,0.0464,0.0327,0.0191,0.0286,0.2212]
a_m12 = [0.9280,0.9446,0.9430,0.9416,0.9002,0.8791]

o_m22 = [0.7117,0.9088,0.9535,0.9523,0.7891]
r_m22 = [0.2179,0.0427,0.0166,0.0153,0.1800]
a_m22 = [0.9296,0.9514,0.9701,0.9676,0.9692]

labels2 = ['p=0','p=0.13','p=0.25','p=0.5','p=0.75','p=1.0']
labelsm12 = ['p=0','p=0.25','p=0.3','p=0.5','p=0.75','p=1.0']
labelsm22 = ['p=0','p=0.25','p=0.5','p=0.75','p=1.0']
#n=2



#n=3
o_cm = [0.6523,0.6962, 0.6974,0.6943,0.6473,0.5847]
r_cm = [0.1290,0.0869, 0.0865,0.0362,0.0272,0.0512]
a_cm = [0.7813,0.7831, 0.7839,0.7305,0.6745,0.6359]

o_m1 = [0.6744,0.8088,0.8643,0.8682 ,0.8463,0.7627]
r_m1 = [0.2837,0.1267,0.0509,0.0497 ,0.0446,0.1085]
a_m1 = [0.9581,0.9356,0.9152,0.9179 ,0.8910,0.8706]

o_m2 = [0.6769,0.8477,0.8966,0.9028 ,0.9012,0.8497]
r_m2 = [0.2758,0.0974,0.0452,0.0418 ,0.0387,0.0909]
a_m2 = [0.9527,0.9451,0.9418,0.9416 ,0.9400,0.9407]

labels = ['p=0','p=0.25','p=0.26','p=0.5','p=0.75','p=1.0']
labelsm1 = ['p=0','p=0.25','p=0.5','p=0.56','p=0.75','p=1.0']
labelsm2 = ['p=0','p=0.25','p=0.5','p=0.55','p=0.75','p=1.0']
#n=3


#n=4
o_cm4 = [0.6849,,,,,]
r_cm4 = [0.1083,,,,,]
a_cm4 = [0.7933,,,,,]

o_m14 = [0.7504,0.8116,0.8948,0.9008,0.8921,0.8408]
r_m14 = [0.2224,0.1487,0.0417,0.03,0.0224,0.0429]
a_m14 = [0.9728,0.9603,0.9366,0.9308,0.9145,0.8838]

o_m24 = [0.7505,0.8258,0.9301,0.9423,0.9404,0.9290]
r_m24 = [0.2202,0.1431,0.0310,0.0177,0.0188,0.0263]
a_m24 = [0.9707,0.9690,0.9611,0.9600,0.9593,0.9553]

labels4 = ['p=0','p=0.25','p=0.3','p=0.5','p=0.75','p=1.0']
labelsm14 = ['p=0','p=0.25','p=0.5','p=0.63','p=0.75','p=1.0']
labelsm24 = ['p=0','p=0.25','p=0.5','p=0.63','p=0.75','p=1.0']
#n=4


plt.figure(figsize=(20,15))
#plot 1:

plt.subplot(3, 3, 1)
plt.plot(labels2,o_cm2,color="red",label="D-Score")
plt.plot(labels2,a_cm2,color="green",label="acc")
plt.plot(labels2,r_cm2,color="blue", label="robust")
plt.title("CM")
plt.legend(fontsize=14)
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

#plot 2:

plt.subplot(3, 3, 2)
plt.plot(labelsm12,o_m12,color="red",label="D-Score")
plt.plot(labelsm12,a_m12,color="green",label="acc")
plt.plot(labelsm12,r_m12,color="blue", label="robust")
plt.title("MMA")
plt.legend(fontsize=14,loc='center left')
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
#plot 2:

plt.subplot(3, 3, 3)
plt.plot(labelsm22,o_m22,color="red",label="D-Score")
plt.plot(labelsm22,a_m22,color="green",label="acc")
plt.plot(labelsm22,r_m22,color="blue", label="robust")
plt.title("MMB")
plt.legend(fontsize=14,loc='center left')
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()



#plot 4:

plt.subplot(3, 3, 4)
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

#plot 5:

plt.subplot(3, 3, 5)
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
#plot 6:

plt.subplot(3, 3, 6)
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


#plot 7:

plt.subplot(3, 3, 7)
plt.plot(labels4,o_cm4,color="red",label="D-Score")
plt.plot(labels4,a_cm4,color="green",label="acc")
plt.plot(labels4,r_cm4,color="blue", label="robust")
plt.title("CM")
plt.legend(fontsize=14)
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()

#plot 8:

plt.subplot(3, 3, 8)
plt.plot(labelsm14,o_m14,color="red",label="D-Score")
plt.plot(labelsm14,a_m14,color="green",label="acc")
plt.plot(labelsm14,r_m14,color="blue", label="robust")
plt.title("MMA")
plt.legend(fontsize=14,loc='center left')
plt.xlabel("probability",fontsize=14)
plt.ylabel("score",fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.grid()
#plot 9:

plt.subplot(3, 3, 9)
plt.plot(labelsm24,o_m24,color="red",label="D-Score")
plt.plot(labelsm24,a_m24,color="green",label="acc")
plt.plot(labelsm24,r_m24,color="blue", label="robust")
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