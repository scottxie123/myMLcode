import numpy as np

class logistic_regression():
    '''
    X_train:n x dim
    y_train: n
    '''
    def __init__(self,X_train,y_train,lr = 1e-2,maxiter = 200,eps = 1e-6):
        assert X_train.shape[0] == len(y_train)
        self.X_train = X_train
        self.y_train = y_train
        self.dim = self.X_train.shape[1]+1
        self.X_train = np.concatenate((self.X_train,np.ones((self.X_train.shape[0],1))),axis =1)
        self.w = np.zeros((1,self.dim))
        self.lr = lr
        self.maxiter = maxiter
        self.eps = eps
    
    def fi(self,x):
        '''
        x: n x dim

        return 1x n
        '''
        return 1/(1+np.exp(-1*self.w.dot(x.T)))
    
    def predict (self,x):
        x = np.concatenate((x,np.ones((x.shape[0],1))),axis =1)
        return self.fi(x)
    
    def diff_fi(self,x,fix):
        '''
        x:n x dim
        fix: 1 x n

        return: n x dim
        '''
        #print(x.shape)
        return (fix**2*np.exp(-1*self.w.dot(x.T))).T*x
    
    def negeitive_gradient_Lw(self):
        cache_fi = self.fi(self.X_train)  #n x 1
        cache_diff_fi = self.diff_fi(self.X_train,cache_fi) #n x dim
        #print(cache_diff_fi.shape)
        return ((self.y_train/(cache_fi+self.eps)-(1-self.y_train)/(1-cache_fi+self.eps)).dot(cache_diff_fi)).T #dim x 1
    
    def fit(self):
        for i in range(self.maxiter):
            #print(i)
            n_gradient = self.negeitive_gradient_Lw().T
            n_gradient = n_gradient/np.linalg.norm(n_gradient)
            self.w += n_gradient*self.lr
            print(self.w)




