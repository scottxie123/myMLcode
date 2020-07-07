import numpy as np
import matplotlib.pyplot as plt
def gen_data():
    x1 = (np.random.rand(1000)-0.5)*4
    x2 = (np.random.rand(1000)-0.5)*4
    y11 = np.sqrt((4-np.square(x1[:500])))
    y12 = -1*np.sqrt((4-np.square(x1[500:])))
    y21 = np.sqrt((4-np.square(x2[:500])))
    y22 = -1*np.sqrt((4-np.square(x2[500:])))
    y1 = np.append(y11,y12)
    y2 = np.append(y21,y22)
    
    x3 = np.random.randn(2,1000)/2
    x4 = np.random.randn(2,1000)/2

    eps1 = np.random.randn(2,1000)/6
    eps2 = np.random.randn(2,1000)/6

    X_train = np.concatenate((np.array([x1,y1])+eps1,x3),axis=1)
    y_train = np.concatenate((np.ones(1000),-1*np.ones(1000)))
    X_test = np.concatenate((np.array([x2,y2])+eps2,x4),axis=1)
    y_test = np.concatenate((np.ones(1000),-1*np.ones(1000)))
    return X_train,y_train,X_test,y_test

def gen_linear_data():
    x1 = np.random.randn(2,1000)+np.array([[2,2]]).T*2
    x2 = np.random.randn(2,1000)#-np.array([[2,2]]).T
    x3 = np.random.randn(2,1000)+np.array([[2,2]]).T*2
    x4 = np.random.randn(2,1000)#-np.array([[2,2]]).T

    X_train = np.concatenate((x1,x2),axis=1)
    y_train = np.concatenate((np.ones(1000),-1*np.ones(1000)))
    X_test = np.concatenate((x3,x4),axis=1)
    y_test = np.concatenate((np.ones(1000),-1*np.ones(1000)))
    return X_train,y_train,X_test,y_test
if __name__ == "__main__":
    a,b,c,d = gen_linear_data()
    print(c.shape)
    plt.scatter(c[0],c[1])
    plt.show()
    