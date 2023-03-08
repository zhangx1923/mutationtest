import numpy as np
def BCE(y_true, y_pred):
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
    term_1 = y_true * np.log(y_pred + 1e-7)
    return -np.mean(term_0+term_1, axis=0)[0]

n = 3
y1 = [1,0,0,0,0,0,0,0,0]
y2 = [1/(n*n)]*9
y3 = y1
y4 = [0.1]*9
y5 = [1]*9
print(BCE(y2,y1), BCE(y2,y1), BCE(y5,y4))