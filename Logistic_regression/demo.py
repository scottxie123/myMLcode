import numpy as np
from generate_data import *
from time import time
import matplotlib.pyplot as plt
from logistic_regression import logistic_regression

print('Loading Data ...')
t = time()
X_train,y_train,X_test,y_test = gen_linear_data()
print("Done!",time()-t)

L_R = logistic_regression(X_train,y_train)
print('Training...')
t = time()
L_R.fit()
print('Done!',time()-t)

y_pre = L_R.predict(X_test)
print(((y_pre>0.5)==y_test)[0])
result = (y_pre>0.5)==y_test
print(np.sum(result)/2000)

plt.scatter(X_train.T[0][:1000],X_train.T[1][:1000])
plt.scatter(X_train.T[0][1000:],X_train.T[1][1000:])
plt.show()
