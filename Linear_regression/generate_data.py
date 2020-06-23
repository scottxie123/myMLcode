import numpy as np 

def  gen_data(num, dim=5,disturb=False):
    a = np.random.rand(dim)
    #b = np.random.reand()
    x = np.random.rand(dim,num)*10 #生成[0,10]之间的随机数
    eps = np.random.randn(num) #扰动
    y = a.dot(x)+eps
    if disturb:
        ind = np.random.choice(range(num))
        y[ind] +=300
    #print(x.shape,y.shape)
    return x,y,a#,b