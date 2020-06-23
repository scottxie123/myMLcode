import numpy as np

def linear_regression(x,y):
   # dim,num = x.shape()
   return np.linalg.inv(x.dot(x.T)).dot(x).dot(y)