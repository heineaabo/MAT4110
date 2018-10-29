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
        u,d,v = np.linalg.svd(img,compute_uv=True)
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
        
