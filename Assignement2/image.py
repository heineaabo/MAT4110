from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

def get(path, weighted=False, rgb=False):
    im = Image.open(path)
    w,h = im.size
    img = np.zeros([h,w])
    if rgb:
        r = np.zeros([h,w])
        g = np.zeros([h,w])
        b = np.zeros([h,w])
    print('Image shape: ', img.shape)
    for i in range(w):
        for j in range(h):
            pixel = im.getpixel((i,j))
            if rgb:
                r[j,i] = pixel[0]
                g[j,i] = pixel[1]
                b[j,i] = pixel[2]
            if weighted:
                img[j,i] = (0.2126*pixel[0] + 0.7152*pixel[1] + 0.0722*pixel[2])
            else:
                img[j,i] = (pixel[0] + pixel[1] + pixel[2])/3
    if rgb:
        return img, r,g,b
    else:
        return img
    
def show(img, cmap='gray', name=''):
    plt.figure()
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    if name != '':
        plt.savefig(name)

def save(img, name):
    new_p = Image.fromarray(img.astype(np.float64))
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
    new_p.save(name)

def get_size(name, prnt=False):
    if prnt:
        print(os.stat(name).st_size)
    return os.stat(name).st_size

def compress(filename, r, plot=False, sigma=False):
    path = './Images/'+filename[:-4]+'/'
    img_name = path+filename[:-4]+'_gray.jpg'
    try: 
        plt.imread(img_name)
        print('Grayscale image found')
        img = Image.open(img_name)
    except FileNotFoundError:
        print('Creating grayscale image ...')
        img = get(path+filename)
        save(img,img_name)
        print('Grayscale image created')
    if filename[:-4].lower() == 'jellyfish':
        name = path + 'jellyfish_gray_r{}.jpg'.format(r)
    elif filename[:-4].lower() == 'chessboard':
        name = path + 'chessboard_gray_r{}.jpg'.format(r)
    elif filename[:-4].lower() == 'new_york':
        name = path + 'new_york_gray_r{}.jpg'.format(r)
    else: raise TypeError('Choose image file from directory')
    try:
        plt.imread(name)
        print('Compressed image (r = {}) found'.format(r))
        compressed = Image.open(name)
        readfile = open(path+filename[:-4]+'.txt', 'r')
        for line in readfile:
            if line[:10] == '{:<10}'.format(r):
                before = line[10:25]
                after = line[25:40]
                ratio = line[40:]
        readfile.close()
        print('Bytes before: ',before)
        print('Bytes after: ',after)
        print('Compression ratio', ratio)
    except FileNotFoundError:
        print('Compressing image ...')
        #u,d,v = np.linalg.svd(img,compute_uv=True)
        u,d,v = svd(np.asarray(img))
        if u.shape[0] > v.shape[0]:
            S = (np.concatenate((np.diag(d), np.zeros((np.abs(u.shape[0]-u.shape[0]), v.shape[0]))), axis=0))
        if u.shape[0] < v.shape[0]:
            S = (np.concatenate((np.diag(d), np.zeros((np.abs(u.shape[0]-v.shape[0]), u.shape[0]))), axis=0)).T
        compressed = u[:,:r]@S[:r]@v
        save(compressed,name=name)
        if sigma:
            plt.figure(figsize = [7,5])
            plt.title('Singular values used', fontsize=18)
            plt.semilogy(np.linspace(0,r,r), (d[:r]), 'k.')
            plt.xlabel(r'x', fontsize=12)
            plt.ylabel(r'log($\sigma_x$)', fontsize=12)
            plt.savefig(path+'sigma_r{}.pdf'.format(r))
        before = get_size(img_name,prnt=False)
        after = get_size(name,prnt=False)
        ratio = get_size(img_name,prnt=False)/get_size(name,prnt=False)
        writefile = open(path+filename[:-4]+'.txt', 'a')
        text = '{:<10}{:<15}{:<15}{:.15}\n'.format(r,before,after,ratio)
        writefile.write(text)
        writefile.close()
        print('Bytes before: ',before)
        print('Bytes after: ',after)
        print('Compression ratio', ratio)
    if plot:
        show(img)
        plt.title('Uncompressed')
        show(compressed)
        plt.title('Compressed r = {}'.format(r))
        
def svd(A, sing_mat=False):
    """ 
    Factorize a matrix A = USV.T
    """
    if A.shape[1] < A.shape[0]:
        A = A.T
    n = A.shape[1]
    m = A.shape[0]
    k = np.abs(n-m)
    d1, u = np.linalg.eigh(A@A.T)
    d2, v = np.linalg.eigh(A.T@A)
    D = -np.sort(-d1) #Get diagonal elements in descending order
    print(D)
    print(min(n,m))
    
    # make i > n-k elements equal to 0 if precision is bad
    #if n != m:
    #    for i in range(k):
    #        D[min(n,m)+i] = 0
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
    print(np.sqrt(D).shape)
    print(np.zeros((np.abs(V.shape[0]-U.shape[0]), Vpri.shape[0])).shape)
    S = (np.concatenate((np.sqrt(D), np.zeros((np.abs(V.shape[0]-U.shape[0]), V.shape[0]))), axis=0))          
#    if sing_mat:
#        S = np.diagflat(np.sqrt(D)) # singular values, sqrt(d) are the diagonal elements of a matrix
#    if sing_mat == False:
#             S = np.sqrt(D)
    if n != m:
        return U, S[:,:min(m,n)], V.T
    else:
        return U, S, V.T