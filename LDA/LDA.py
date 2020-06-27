import numpy as np
from generate_data import gen_data
def LDA(data1,data2):
    '''
    线性判别分析
    input:data1,data2:训练集
    output: w(直线方向),mu1(均值1),mu2(均值2)
    '''
    mu1 = np.mean(data1,0)
    mu2 = np.mean(data2,0)
    Sigma1 = (data1-mu1).T.dot((data1-mu1))
    Sigma2 = (data2-mu1).T.dot((data2-mu2))
    Sw = Sigma1+Sigma2
    u,s,v = np.linalg.svd(Sw,0)
    return v.dot(np.diag(1/s)).dot(u.T).dot(mu1-mu2),mu1,mu2

if __name__ == '__main__':
    d1,d2,c1,c2 = gen_data(1000)
    w,_,_ = LDA(d1,d2)
    print(w/np.linalg.norm(w))
    w_ = c1-c2
    print(w_/np.linalg.norm(w_))
