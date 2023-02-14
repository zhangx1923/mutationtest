import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
import numpy as np
from sympy import hn1, hn2
# plt.rcParams['axes.unicode_minus'] = False
# plt.rcParams['axes.facecolor'] = '#cc00ff'
# plt.rcParams['font.sans-serif'] = ['STKAITI']
# 创建画布
fig = plt.figure()
# 创建3D坐标系
axes3d = Axes3D(fig)
zs = range(3)
left = np.arange(0, 3)
height = np.array([])
h = [[0.01,0.04,0.04],[0.14,0.39,0.22],[0.05,0.08,0.03]]

f1 = [[0.55,12.04,1.21],[4.77,22.67,2.02],[1.05,7.3,0.76]]

f2 = [[0.07,3.44,0.49],[1.16,19.65,3.66],[0.15,10.92,0.48]]

f3 = [[2.31,3.9,1.91],[3.57,8.62,3.51],[2.58,4.1,2.06]]

a1 = [[21.65,36.56,18.66],[45.23,94.36,43.54],[28.51,39.65,11.03]]

a2 = [[21.03,41.85,29.63],[56.20,96.39,62.80],[30.01,42.18,10.03]]

a3 = [[42.31,48.61,40.90],[51.22,62.40,50.88],[44.29,55.11,45.28]]

def calNorm(a):
    sum = 0
    for row in a:
        for j in row:
            sum += j
    for i in range(0,len(a)):
        for j in range(0,len(a[i])):
            a[i][j] = a[i][j] / sum
    return a

h1 = calNorm(a3)
print(h1)
for i in range(len(zs)):
    z = zs[i]
    np.random.seed(i)
    height = np.array(h1[i])
    axes3d.bar(left, height, zs=z, zdir='x',
               color=['green', 'blue', 'orange', 'purple', 'red', 'black', 'gray', 'orange', 'green', 'cyan'])
plt.xticks(zs, ['1', '2', '3'])
plt.yticks(left, ['1','2','3'])
plt.xlabel('X')
plt.ylabel('Y')
plt.show() 


# # surface.py

# import matplotlib.pyplot as plt
# import numpy as np

# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# x = np.arange(0, 3, 1)
# y = np.arange(0, 3, 1)
# X, Y = np.meshgrid(x, y)

# Z = np.array([[0.06,0.07,0.01],[0.15,0.32,0.20],[0.04,0.09,0.06]])
# print(Z)

# surf = ax.plot_surface(X, Y, Z, cmap=cm.cool)
# fig.colorbar(surf, shrink=0.05, aspect=1)

# plt.show()


# wireframe.py

# import matplotlib.pyplot as plt
# import numpy as np

# from mpl_toolkits.mplot3d import Axes3D

# fig = plt.figure()
# ax = fig.gca(projection='3d')

# x = np.arange(0, 3, 1)
# y = np.arange(0, 3, 1)
# X, Y = np.meshgrid(x, y)

# Z = np.array([[0.06,0.07,0.01],[0.15,0.32,0.20],[0.04,0.09,0.06]])

# surf = ax.plot_wireframe(X, Y, Z)

# plt.show()