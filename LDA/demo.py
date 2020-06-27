import numpy as np
from generate_data import gen_data
from LDA import LDA
import matplotlib.pyplot as plt

d1,d2,c1,c2 = gen_data(10)
w,mu1,mu2 = LDA(d1,d2)
w = w/np.linalg.norm(w)
#print(np.linalg.norm(w))
a = w[1]/w[0]



plt.figure(1)
plt.axis('equal')
x = np.linspace(-8,8,100)
y = a*x
plt.plot(x,y)

for x1,wx1 in zip(d1,d1.dot(w)):
    #print(wx1)
    y1 = wx1*w
    plt.scatter(x1[0],x1[1],color='r')
    plt.scatter(y1[0],y1[1],color='g')
    plt.plot([x1[0],y1[0]],[x1[1],y1[1]],color = 'y',linestyle = '--')

for x1,wx1 in zip(d2,d2.dot(w)):
    #print(wx1)
    y1 = wx1*w
    plt.scatter(x1[0],x1[1],color='b')
    plt.scatter(y1[0],y1[1],color='skyblue')
    plt.plot([x1[0],y1[0]],[x1[1],y1[1]],color = 'y',linestyle = '--')

plt.scatter(mu1[0],mu1[1],color = 'k')
plt.scatter(mu2[0],mu2[1],color = 'k')
plt.plot([mu1[0],mu2[0]],[mu1[1],mu2[1]],color = 'k')

plt.show()