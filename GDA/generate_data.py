import numpy as np

def gen_data():
    x1 = np.random.randn(2,1000)+4
    x2 = np.random.randn(2,100)#-np.array([[2,2]]).T
    x3 = np.random.randn(2,1000)+4
    x4 = np.random.randn(2,100)#-np.array([[2,2]]).T

    X_train = np.concatenate((x1,x2),axis=1)
    y_train = np.concatenate((np.ones(1000)*0,np.ones(100)))
    X_test = np.concatenate((x3,x4),axis=1)
    y_test = np.concatenate((np.ones(1000)*0,np.ones(100)))
    return X_train,y_train,X_test,y_test

if __name__ == "__main__":
    a,b,c,d = gen_data()
    print(a.shape,b.shape,c.shape,d.shape == 2000)