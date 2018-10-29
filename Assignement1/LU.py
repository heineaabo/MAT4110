import numpy as np
import scipy.linalg as la

# LU factorization of a non-singular square matrix A (nxn)

def factorize(A):
    n = A.shape[0]
    L = np.zeros((n,n))
    U = np.zeros((n,n))
    
    #print('B')
    #print(B)
 
    # First elements
    U[0] = A[0]
    L.T[0] = A.T[0]/A[0][0]
    #print('u1 = ', U[0])
    #print('l1 = ', L.T[0])
    Ak = A - np.outer(L.T[0],U[0])  
    for i in range(1,n):
        print('iteration: ',i)
        print('u',1,' = ', U[i-1])
        print('l',1,' = ', L.T[i-1])
        print('LU')
        print(np.outer(L.T[i-1],U[i-1]))
        print('L')
        print(L)
        print('U')
        print(U)
        print('A',i,' = ')
        print(Ak)
        L.T[i] = Ak.T[i]/Ak[i][i]
        U[i] = Ak[i]
        print('---------------')
        Ak -= np.outer(L.T[i],U[i])
    return(L,U)

B = np.array([[3, 2, 1],[1, 2, 0],[2, 3, 1]])
l,u = factorize(B)
print('DONE!!')
print('L')
print(l)
print('U')
print(u)

(P, L1, U1) = la.lu(B)
print(L1)
print(U1)
