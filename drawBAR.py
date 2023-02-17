import numpy as np
import matplotlib.pyplot as plt

labels = []
overall = []
robust = []
acc = []

barWidth = 0.25
r1 = np.arrange(len(overall))
r2 = [x+ barWidth for x in r1]
r3 = [x + barWidth for x in r2]

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.08, 1.02*height, "%s" % int(height), size=10)

plt.figure(figsize=(6,4))
cm = plt.bar(r1,overall,color="red",width=barWidth,edgecolor='white',label='overall')
autolabel(cm)
cm = plt.bar(r2,robust,color="blue",width=barWidth,edgecolor='white',label='robust')
autolabel(cm)
cm = plt.bar(r3,acc,color="green",width=barWidth,edgecolor='white',label='acc')
autolabel(cm)

plt.ylim((0,100))
plt.xlabel("score type")
plt.ylabel("score value")
plt.yticks(size=10)
plt.xticks([r+barWidth for r in range(len(overall))],labels=labels, size=10)

#plt.legend()
plt.savefig("bar.jpg")