import numpy as np
from generate_data import gen_data
from GDA import GDA
from time import time

print('Loading data...')
t = time()
X_train,y_train,X_test,y_test = gen_data()
print("Done! ",time()-t)
print("Training...")
t = time()
model = GDA(X_train,y_train)
model.fit()
print("Done! ",time()-t)
pre = model.predict(X_test)
result = pre==y_test
print(sum(result)/len(result))