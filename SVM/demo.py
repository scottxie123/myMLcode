# 读入癌症数据集
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from SVM import SVM
from kernal import *
from time import time

print('Loading Data ...')
t = time()
cancer=datasets.load_breast_cancer()
X=cancer.data
X = (X-X.min())/(X.max()-X.min())
y=(cancer.target-0.5)*2

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3, random_state=42)

X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print("Done!",time()-t)

svm = SVM(X_train,y_train,C = 1,toler = 0.001,kernal = gaussian_kernal,parameter=[0.1])
print('Training...')
t = time()
svm.fit()
print('Done!',time()-t)
results = []
for i in range(X_test.shape[0]):
    y_pre = svm.predict(X_test[i])
    results.append(2.0*((y_pre>0)-0.5)==y_test[i])
results = np.array(results)
print(np.sum(results)/len(results))
#print(svm.alpha)
