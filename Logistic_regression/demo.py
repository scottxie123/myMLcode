import numpy as np
from generate_data import *
from time import time
import matplotlib.pyplot as plt
from logistic_regression import logistic_regression

print('Loading Data ...')
t = time()
X_train,y_train,X_test,y_test = gen_linear_data()
print("Done!",time()-t)

L_R = logistic_regression(X_train,y_train,maxiter=4000)
print('Training...')
t = time()
L_R.fit()
print('Done!',time()-t)

y_pre = L_R.predict(X_test)
print(((y_pre>0.5)==y_test)[0])
result = (y_pre>0.5)==y_test
print(np.sum(result)/2000)

x_1 = np.linspace(-5,6,num=200)
y_1 = np.linspace(-5,6,num=200)
[xx,yy] = np.meshgrid(x_1,y_1)
xx = xx.reshape(-1)
yy = yy.reshape(-1)
result = []
for x_2,y_2 in zip(xx,yy):
    pre = L_R.predict(np.array([x_2,y_2]).reshape(1,-1))>0.5
    result.append(pre)
ind = np.nonzero(result)[0]
ind2 = np.nonzero([not i for i in result])[0]

plt.scatter(X_train.T[0][:1000],X_train.T[1][:1000])
plt.scatter(X_train.T[0][1000:],X_train.T[1][1000:])
plt.scatter(xx[ind],yy[ind],c = 'b',alpha=0.1, lw=0)
plt.scatter(xx[ind2],yy[ind2],c = 'g',alpha=0.1, lw=0)
plt.show()
