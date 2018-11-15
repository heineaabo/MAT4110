import numpy as np
import scipy.linalg as la

###################################
###### FACTORIZATION METHODS ######
###################################

# SVD (factorization)
def svd(A):
    """ 
    Factorize a matrix A = USV.T
    """
    n = A.shape[1]
    m = A.shape[0]
    k = np.abs(n-m)
    d1, u = np.linalg.eigh(A@A.T)
    d2, v = np.linalg.eigh(A.T@A)
    D = -np.sort(-d1) #Get diagonal elements in descending order
    
    # make i > n-k elements equal to 0 if precision is bad
    if n != m:
        for i in range(k):
            D[min(n,m)+i] = 0
    # Need to sort u in the order of 
    # descending sort of eigenvalues d1
    U = np.zeros_like(u)
    for i in range(u.shape[0]):
        if n != m and i > max(n,m)-k: #For nxm matrices
            U.T[i] = u.T[np.argsort(-d1)[i-max(n,m)-k]]
        U.T[i] = u.T[np.argsort(-d1)[i]]
                
    # Need to sort v in the order of 
    # descending sort of eigenvalues d2        
    V = np.zeros_like(v)
    for i in range(v.shape[0]):
        V.T[i] = v.T[np.argsort(-d2)[i]]
                
    S = np.diagflat(np.sqrt(D)) # singular values, sqrt(d) are the diagonal elements of a matrix
    
    if n != m:
        return U, S[:,:min(m,n)], V
    else:
        return U, S, V

# Cholesky factorization 
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

# LU factorization    
def LU(A):
    n = A.shape[0]
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n): 
        U[i] = A[i]
        L.T[i] = A.T[i] / A[i][i]
        A = A - np.outer(L.T[i],U[i])
    return(L,U)

# QR factorization
def QR(A):
    """For square matrix A"""
    n,m = A.shape
    if n > m:
        A = A.T 
    n = A.shape[0]
    Q = np.zeros((n,m))
    R = np.zeros((n,m))
    for k in range(n):
        w = A.T[k]
        for i in range(k):
            w = w - np.inner(Q.T[i],A.T[k])*Q.T[i]
            R[i][k] = np.inner(Q.T[i],A.T[k])
        R[k][k] = np.sqrt(np.inner(w,w))
        Q.T[k] = w / R[k][k]
    return (Q,R)

####################################
###### LEAST SQUARE SOLUTIONS ######
####################################

# Least square problem with Cholesky factorization
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


# Least square problem with QR factorization
def least_qr(A,b):
    """
    R = [R1,0].T
    Q.T@y = [c1,c2].T
    R1x = c1
    """
    Q,R = np.linalg.qr(A,mode='reduced')
    R1 = R[:m]
    Qb = Q.T @ b
    Q1 = Qb[:m]
    x = back(R1,Q1)
    return x

####################################
####### MATRIX CALCULATIONS ########
####################################

# Forward substitution
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

# Back substitution
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

####################################
########## Interpolation ###########
####################################

def Lagrange_interpole(F,num):
    """
    Take F on shape (x,y)_i (can use np.column_stack):
    f = np.array([(1,3),(3,4),(5,6),(7,-10)])
        -> F = [[1,  3],
                [3,  4],
                [5,  6],
                [7,-10]]
            
    num = number of points to interpolate
    Return: X -> x points
            P -> y points
    """
    def L(f,x,k): # Basis
        B = 1.0
        #i = - 1
        for j in range(f.shape[0]):
            if j != k:
                B *= (x-X[j]) / (X[k]-X[j])
        return B
    fx,f = F.T[0],F.T[1]
    X = np.linspace(fx[0],fx[-1],num)
    P = np.zeros_like(X)
    for i,x in enumerate(X):
        for k,f_k in enumerate(f):
            P[i] += f_k*lagrange(fx,x,k)
    return X,P