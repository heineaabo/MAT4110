import numpy as np
from random import random
import matplotlib.pyplot as plt
 
def forward(L,b):
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0]/L[0][0]
    for i in range(1,n):
        x[i] = b[i]
        for j in range(i):
            x[i] -= (L[i][j]*x[j])
        x[i] = x[i]/L[i][i]
    return x

def back(U,b):
    n = U.shape[0]
    x = np.zeros(n)
    if b.shape == (1,n):
        b = b.T
    x[n-1] = b[n-1]/U[n-1][n-1]
    for i in range(n-1,-1,-1):
        x[i] = b[i]
        for j in range(i+1,m):
            x[i] -= (U[i][j]*x[j])
        x[i] = x[i]/U[i][i]
    return x

def cholesky(A, dhalf = False):
    n = A.shape[0]
    L = np.zeros((n,n))
    D = np.zeros((n,n))
    if dhalf:
        D2 = np.zeros((n,n))
    for i in range(n):
        D[i][i] = A[i][i]
        if dhalf:
            D2[i][i] = np.sqrt(A[i][i])
        L.T[i] = A.T[i] / A[i][i]
        A = A - D[i][i]*np.outer(L.T[i],L.T[i])
    if dhalf:
        return (L,D,D2)
    else:
        return(L,D)

### FUNCTIONS FOR SOLVING NORMAL EQUATIONS
# CHOLESKY
def least_cholesky(B,Ab):
    """
    (A.T@A)x = A.T@b
      =>  Bx = y
  => R.T@R@x = y
    => R.T@w = y
     where w = R@x
     
     R   - upper triangular
     R.T - lower triangular 
    """
    L,D,D2 = cholesky(B,True)
    R = L@D2
    w = forward(R,Ab)
    x = back(R.T,w)
    
    return x

    
# QR
def least_qr(A,b):
    """
    R = [R1,0].T
    Q.T@y = [c1,c2].T
    R1x = c1
    """
    Q,R = np.linalg.qr(A,mode='reduced')
    #Qm,Rm = QR(A)
    #print(Q.shape)
    #print(Qm.shape)
    R1 = R[:m]
    Qb = Q.T @ b
    Q1 = Qb[:m]
    x = back(R1,Q1)
    
    return x

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

# find the A matrix
m = 3
A = np.array([x**i for i in range(m)]).T
B = A.T@A
Ab1 = A.T@y1
Ab2 = A.T@y2

# QR
xhat1 = least_qr(A,y1)
yp1 = A.dot(xhat1)

xhat2 = least_qr(A,y2)
yp2 = A.dot(xhat2)

# Cholesky
xhat3 = least_cholesky(B,Ab1)
yp3 = A.dot(xhat3)

xhat4 = least_cholesky(B,Ab2)
yp4 = A.dot(xhat4)

#PLOT
#qr set 1
plt.subplot(221)
plt.plot(x,y1, 'b.')
plt.plot(x,yp1, 'r-')
#plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'QR Eq.(2)')
plt.xlim(-2,2)

#cholesky set 1
plt.subplot(222)
plt.plot(x,y1, 'b.')
plt.plot(x,yp3, 'r-')
#plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Cholesky Eq.(2)')
plt.xlim(-2,2)

#qr set 2
plt.subplot(223)
plt.plot(x,y2, 'b.')
plt.plot(x,yp2, 'r-')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'QR Eq.(3)')
plt.xlim(-2,2)

#cholesky set 2
plt.subplot(224)
plt.plot(x,y2, 'b.')
plt.plot(x,yp4, 'r-')
plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'Cholesky Eq.(3)')
plt.xlim(-2,2)

plt.show()

