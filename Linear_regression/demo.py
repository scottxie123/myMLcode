import numpy as np
from matplotlib import pyplot  as plt
from generate_data import gen_data
from L_r import linear_regression

#计算高维线性回归并计算误差
x,y,w = gen_data(200, 5)
w_ = linear_regression(x,y)
print(w- w_)

#计算1维情况并作图
x,y,w = gen_data(200,1)
#x,y,w = gen_data(200,1,1) #加入不合理的点
w_ = linear_regression(x,y)
x1 = np.arange(0,10,0.1).reshape(1,-1)
y1 = w.dot(x1)
y2 = w_.dot(x1)
plt.figure(1)
plt.plot(x.reshape(-1),y,'r+')
plt.plot(x1.reshape(-1),y1,'b',x1.reshape(-1),y2,'g')
plt.show()