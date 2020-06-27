import numpy as np
#center1 = np.array([0,0])
#center2 = np.array([3,5])

def gen_data(num=100,dim=2):
    '''
        两个中心点，分别生成num个随机点
        input num,dim
        output data1(num*2),data2(num*2)
    '''
    center1 = np.random.rand(dim)
    center1[0] -= 5
    center2 = np.random.rand(dim)
    center2[0] += 5
    data1 = np.random.randn(num,dim)+center1
    data2 = np.random.randn(num,dim)+center2
    return data1,data2,center1,center2


if __name__ == '__main__':
    data1,data2,c1,c2 = gen_data(15,3)
    print(data1.shape,data1)
    print(c1,c2)