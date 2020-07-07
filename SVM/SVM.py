import numpy as np
from kernal import linear_kernal

class  SVM():
    def __init__(self,train_data,train_label,C,toler,kernal=linear_kernal, parameter = [],maxiter=5,eps = 1e-10):
        #模型参数
        self.C = C
        self.toler = toler
        self.kernal = kernal
        self.parameter = parameter
        self.maxiter = maxiter
        self.eps = eps

        #训练数据加载
        self.train_data = train_data #[n,dim]
        self.train_label = train_label.reshape(-1) #[dim]
        self.n = train_data.shape[0]
        self.dim = train_data.shape[1]
        #print(self.dim,len(train_label))
        assert self.n == len(train_label)

        #变量初始化
        self.E = np.zeros(self.n) 
        self.alpha = np.zeros(self.n)
        self.b = 0

    def dot(self,x,y):
        '''
        定义内积
        '''
        return self.kernal(x,y,self.parameter)

    def compute_Ek(self,k):
        '''
        计算第k个数据的误差
        '''
        return self.predict(self.train_data[k])-self.train_label[k]

    def predict(self,x):
        ind = np.nonzero(self.alpha)[0]
        if len(ind) == 0:
            return self.b
        return (self.alpha[ind]*self.dot(self.train_data[ind],x)).dot(self.train_label[ind])-self.b

    def update_E(self):
        for i in range(self.n):
            self.E[i] = self.compute_Ek(i)
    
    def clip_alpha(self,alpha,L,H):
        if alpha>H:
            alpha = H
        if alpha<L:
            alpha = L
        return alpha

    
    def fit(self):
        iter = 0;entire=1;changed_pairs=0
        self.update_E() #更新E
        while (iter<self.maxiter) and (entire or changed_pairs>0):
            changed_pairs = 0
            if entire:
                for i in range(self.n):
                    changed_pairs += self.inner(i)
            else:
                boundind = np.nonzero((self.alpha>0)*(self.alpha<self.C))[0]
                for i in boundind:
                    changed_pairs += self.inner(i)
            iter +=1
            #print(entire)
            entire = not entire
            print('iter:',iter,' changed pairs: ',changed_pairs)


    def inner(self,i):
        #输入的数据i要求不满足KKT条件，然后选择数据j，分别更新alphai,alphaj
        #print(self.train_label[i]*self.E[i]<-self.toler,self.alpha[i]<self.C,self.train_label[i]*self.E[i]>self.toler,self.alpha[i]>0)
        if (self.train_label[i]*self.E[i]<-self.toler and self.alpha[i]<self.C) or (self.train_label[i]*self.E[i]>self.toler and self.alpha[i]>0):
            delta_E = np.abs(self.E-self.E[i])
            j = np.argmax(delta_E)#选择delta_E最大的j

            #compute L H
            if self.train_label[i] != self.train_label[j]:
                L = max(0,self.alpha[i]-self.alpha[j])
                H = min(self.C,self.C+self.alpha[i]-self.alpha[j])
            else:
                L = max(0,self.alpha[j]+self.alpha[i]-self.C)
                H = min(self.C,self.alpha[j]+self.alpha[i])
            if L==H:
                return 0
            
            eta = self.dot(self.train_data[i],self.train_data[i])+self.dot(self.train_data[j],self.train_data[j])-2.0*self.dot(self.train_data[i],self.train_data[j])
            if eta == 0:
                print('eta=0')
                return 0
            
            alphaInew = self.clip_alpha(self.alpha[i]+self.train_label[i]*(self.E[j]-self.E[i])/eta,L,H)
            if np.abs(alphaInew-self.alpha[i])<self.eps*(alphaInew+self.alpha[i]+self.eps):
                #print('too small')
                return 0
            
            alphaJold = self.alpha[j]
            alphaIold = self.alpha[i]
            self.alpha[i] = alphaInew
            self.alpha[j] += self.train_label[j]*self.train_label[i]*(alphaIold-alphaInew)
            #print(alphaJold,self.alpha[j])
            #更新Ei,Ej
            #self.E[i] = self.compute_Ek(i)
            #self.E[j] = self.compute_Ek(j)
            #更新b
            b1 = self.E[i]+self.train_label[i]*(self.alpha[i]-alphaIold)*self.dot(self.train_data[i],self.train_data[i])\
                +self.train_label[j]*(self.alpha[j]-alphaJold)*self.dot(self.train_data[i],self.train_data[j])+self.b
            b2 = self.E[j]+self.train_label[i]*(self.alpha[i]-alphaIold)*self.dot(self.train_data[i],self.train_data[j])\
                +self.train_label[j]*(self.alpha[j]-alphaJold)*self.dot(self.train_data[j],self.train_data[j])+self.b
            
            if self.alpha[i]>0 and self.alpha[i]<self.C:
                self.b = b2
            elif self.alpha[j]>0 and self.alpha[j]<self.C:
                self.b = b1
            else:
                self.b = (b1+b2)/2.0
            #self.b = (b1+b2)/2.0
            self.update_E()
            #print(np.nonzero(self.alpha),i,j)
            return 1

        else:
            return 0
        
    def compute_w(self):
        print('Warning: 仅当使用线性核时，w才有意义')
        ind = np.nonzero(self.alpha)[0]
        if len(ind) == 0:
            return 0
        return (self.alpha[ind]*self.train_label[ind]).dot(self.train_data[ind])
    
    


    