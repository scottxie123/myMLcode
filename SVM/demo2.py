import numpy as np
from SVM import SVM
from kernal import *
from generate_data import *
from time import time
import matplotlib.pyplot as plt

print('Loading Data ...')
t = time()
X_train,y_train,X_test,y_test = gen_data()
print("Done!",time()-t)

svm = SVM(X_train.T,y_train,C = 1,toler = 0.001,kernal = laplacian_kernal,parameter=[1])
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

x = np.linspace(-4,4)
y = -1*(w[0]*x-svm.b)/w[1]
plt.scatter(X_train[0][:1000],X_train[1][:1000])
plt.scatter(X_train[0][1000:],X_train[1][1000:])
for i in np.nonzero(svm.alpha)[0]:
    plt.scatter(X_train[0][i],X_train[1][i],marker='+',c='k')
plt.plot(x,y)
plt.show()
