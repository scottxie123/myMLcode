import numpy as np
from SVM import SVM
from kernel import *
from generate_data import *
from time import time
import matplotlib.pyplot as plt

print('Loading Data ...')
t = time()
X_train,y_train,X_test,y_test = gen_data()
print("Done!",time()-t)

svm = SVM(X_train.T,y_train,C = 1,toler = 0.001,kernel = laplacian_kernel,parameter=[0.1])
print('Training...')
t = time()
svm.fit()
print('Done!',time()-t)
w = svm.compute_w()
print('w: ',w)
print('b: ',svm.b)
ind = np.nonzero(svm.alpha)[0]
print('id of support vecctor: ',ind)
print('alpha is: ',svm.alpha[ind])
print('label is: ',y_train[ind])
results = []
for i in range(X_test.T.shape[0]):
    y_pre = svm.predict(X_test.T[i])
    results.append(((y_pre>0)*2-1)==y_test[i])
results = np.array(results)
print('accuracy: ',np.sum(results)/len(results))

x = np.linspace(-2,5)
y = -1*(w[0]*x-svm.b)/w[1]
plt.figure()
plt.scatter(X_train[0][:1000],X_train[1][:1000],alpha=0.4, lw=0)
plt.scatter(X_train[0][1000:],X_train[1][1000:],alpha=0.4, lw=0)
for i in np.nonzero(svm.alpha)[0]:
    plt.scatter(X_train[0][i],X_train[1][i],marker='+',c='k')
plt.plot(x,y)
#plt.show()


x_1 = np.linspace(-3.5,8,num=200)
y_1 = np.linspace(-3.5,8,num=200)
[xx,yy] = np.meshgrid(x_1,y_1)
xx = xx.reshape(-1)
yy = yy.reshape(-1)
result = []
for x_2,y_2 in zip(xx,yy):
    pre = svm.predict(np.array([x_2,y_2]))>0
    result.append(pre)
ind = np.nonzero(result)[0]
ind2 = np.nonzero([not i for i in result])[0]
#print(ind)
plt.figure()
plt.plot(x,y,c='k')
plt.scatter(X_test[0][:1000],X_test[1][:1000],marker='+',c='k')
plt.scatter(X_test[0][1000:],X_test[1][1000:],marker='*',c='r')
plt.scatter(xx[ind],yy[ind],c = 'b',alpha=0.1, lw=0)
plt.scatter(xx[ind2],yy[ind2],c = 'g',alpha=0.1, lw=0)
plt.show()