import numpy as np
#x : n*dim
#y : dim
#return : n
def linear_kernel(x,y,parameter = []):
    assert len(parameter) == 0
    return x.dot(y)

def ploy_kernel(x,y,parameter = [1]):
    assert len(parameter) == 1 and parameter[0]>=1
    return np.power(x.dot(y),parameter[0])

def gaussian_kernel(x,y,parameter = [1]):
    assert len(parameter) == 1 and parameter[0]>0
    return np.exp(-np.linalg.norm(x-y,axis=(x-y).ndim-1)**2/2/parameter[0]**2)

def laplacian_kernel(x,y,parameter = [1]):
    assert len(parameter) == 1 and parameter[0]>0
    return np.exp(-np.linalg.norm(x-y,axis=(x-y).ndim-1)/parameter[0])

def sigmoid_kernel(x,y,parameter = [1,-1]):
    assert len(parameter) == 2 and parameter[0] >0 and parameter[1] <0
    return np.tanh(parameter[0]*x.dot(y)+parameter[1])


if __name__ == "__main__":
    a = np.ones((7,6))
    b = np.ones(6)
    print(gaussian_kernel(a,b).shape)
    print(linear_kernel(a,b))