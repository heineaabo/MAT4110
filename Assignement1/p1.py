import numpy as np
from random import random
import matplotlib.pyplot as plt
from substitution import back
from factorize import QR, cholesky

N = 30

start = -2
stop = 2
noise = 1 

x = np.linspace(start,stop,N)

##### Find y-values for both datasets
n = x.shape[0]
y1 = np.zeros(n)
y2 = np.zeros(n)
for i in range(n):
    r = random() * noise
    y1[i] = x[i] * (np.cos(r + 0.5*x[i]**3) + np.sin(0.5*x[i]**3))
    y2[i] = 4*x[i]**5 - 5*x[i]**4 - 20*x[i]**3 + 10*x[i]**2 + 40*x[i] + 10 + r

##### Plot datasets 
plt.figure(1,(10,4))
plt.subplot(121) 
plt.plot(x,y1,'m.')
plt.title('Dataset 1')
plt.subplot(122)
plt.plot(x,y2,'m.')
plt.title('Dataset 2')

m = 4
A = np.array([x**i for i in range(m)]).T
print(A.shape)
Q, R = np.linalg.qr(A)
#q,r = QR(A)
#print(Q)
#print(q)
c1 = Q[:m]@y1[:m]
c2 = Q[:m]@y2[:m]
#print(R)
#print(c1)
#print(c2)
x1 = back(R[:m],c1)
#x3 = back(R.T[:m],c1)
x2 = back(R[:m],c2)
print(x1)
print(x2)
pred1 = A@x1
pred2 = A@x2

## 2
#B = A@A.T
#L,D,D2 = cholesky(B,True)
#R = L.T @ D2.T
#A = R.T@R
#print(y1.shape)
x_1 = np.linalg.inv(A.T@A)@A.T@y1
pred_1 = A@x_1
x_2 = np.linalg.inv(A.T@A)@A.T@y2
pred_2 = A@x_2
plt.subplot(121)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,pred1,'r-')
plt.plot(x,pred_1,'b-')
plt.subplot(122)
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,pred2,'r-')
plt.plot(x,pred_2,'b-')
plt.show()
