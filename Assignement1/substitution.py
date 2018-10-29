import numpy as np

# L or U must be traingular
# L and U are nxn matrices, b is n vector

def forward(L,b):
    n = L.shape[0]
    x = np.zeros(n)
    x[0] = b[0]/L[0][0]
    for i in range(1,n):
        x[i] = b[i]/L[i][i]
        for j in range(i):
            x[i] -= (L[i][j]*x[j])/L[i][i] 
    return x

def back(U,b):
    n = U.shape[0]
    x = np.zeros(n)
    if b.shape == (1,n):
        b = b.T
    x[n-1] = b[n-1]/U[n-1][n-1]
    for i in range(n-1,0,-1):
        x[i] = b[i]/U[i][i]
        for j in range(i):
            x[i] -= (U[i][j]*x[j])/U[i][i]
    return x

#A = np.array([[2.3,3.5,1.2],[0,1.4,2.8],[0,0,2.1]])
#b = np.array([[1.1,2.2,3.3]])

