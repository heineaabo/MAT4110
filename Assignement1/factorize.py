import numpy as np
import scipy.linalg as la

def LU(A):
    n = A.shape[0]
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    for i in range(n): 
        U[i] = A[i]
        L.T[i] = A.T[i] / A[i][i]
        A = A - np.outer(L.T[i],U[i])

    return(L,U)

#X = np.array([[3,2,1],[1,2,0],[2,3,1]])
#l,u = LU(X)

def cholesky(A, dhalf = False):
    """A is a symmetric nxn matrix,
       returns lower triangular nxn L 
       and diagonal nxn D""" 
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

#X = np.array([[3,4],[4,6]])
#l,d,d2=cholesky(X,True)

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
    print('QR')
    print(Q)
    print(R)
    
    return (Q,R)

#X = np.array([[2,1,-3],[0,0,-1],[0,1,4]])
#q,r = QR(X)
