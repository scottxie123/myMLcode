import numpy as np

class GDA():
    def __init__(self,X_train,y_train):
        '''
        X_train.shape = [dim,n]
        y_train.shape = [n]
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.dim,self.n = self.X_train.shape
        assert self.n == self.y_train.shape[0]
        self.Sigema = np.zeros((self.dim,self.dim))
        self.mu_0 = np.zeros((self.dim,1))
        self.mu_1 = np.zeros((self.dim,1))
        self.phi = 0
        self.class_one_ind = np.nonzero(self.y_train)[0]
        self.class_zero_ind = np.nonzero(self.y_train-1)[0]
        self.N1 = len(self.class_one_ind)
        self.N0 = len(self.class_zero_ind)
        self.N = self.N0+self.N1

    def fit(self):
        self.phi = self.N1/self.N
        class_one = self.X_train[:,self.class_one_ind]
        class_zero = self.X_train[:,self.class_zero_ind]
        self.mu1 = np.mean(class_one,axis = 1).reshape(-1,1)
        self.mu0 = np.mean(class_zero,axis = 1).reshape(-1,1)
        self.Mu = self.X_train
        self.Mu[:,self.class_zero_ind] -= self.mu0
        self.Mu[:,self.class_one_ind] -= self.mu1
        self.Sigema = self.Mu.dot(self.Mu.T)/self.n
        self.Sigema_1 = np.linalg.inv(self.Sigema)
        self.det = np.linalg.det(self.Sigema)
        
    
    def p_x_y0(self,X):
        return np.exp(-0.5*np.ones(self.dim).dot(self.Sigema_1.dot(X-self.mu0)*(X-self.mu0) ))/((2*np.pi)**(self.dim/2)*self.det**0.5)
    def p_x_y1(self,X):
        return np.exp(-0.5*np.ones(self.dim).dot(self.Sigema_1.dot(X-self.mu1)*(X-self.mu1) ))/((2*np.pi)**(self.dim/2)*self.det**0.5)

    def predict(self,X):
        p_y0_x = (1-self.phi)*self.p_x_y0(X) #(m,)
        p_y1_x = self.phi*self.p_x_y1(X) #(m,)
        return (p_y1_x>p_y0_x)*1.0


