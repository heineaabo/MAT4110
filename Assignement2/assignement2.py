import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def get(path, weighted=False, rgb=False):
    im = Image.open(path)
    w,h = im.size
    img = np.zeros([h,w])
    if rgb:
        r = np.zeros([h,w])
        g = np.zeros([h,w])
        b = np.zeros([h,w])
    #print('Image shape: ', img.shape)
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
    
def save(img, name):
    new_p = Image.fromarray(img.astype(np.float64))
    if new_p.mode != 'L':
        new_p = new_p.convert('L')
    new_p.save(name)
    
def compress(img, r, name):
    u,s,v = np.linalg.svd(img)
    image.save(u[:,:r]@np.diag(s[:r])@v[:r],name=name)
    print('Size uncomp: ',img.shape[0]*img.shape[1])
    print('Size comp: ',u[:,:r].nbytes+s[:r].nbytes+v[:r].nbytes)
    print('Compression ratio: ',img.shape[0]*img.shape[1]/(u[:,:r].nbytes+s[:r].nbytes+v[:r].nbytes))

def find_rmax(filename):
    path = './Images/'+filename[:-4]+'/'
    img_name = path+filename
    img = image.get(img_name,weighted=True)
    m,n = img.shape
    rmax = round(m*n/(m+n+1*10))-1
    compressed,u,s,v = size_svd(img,rmax, filename[:-4])
    ratio = img.nbytes/compressed
    print('Image: ',path)
    print('rmin = ',rmax)
    print('Uncompressed grayscale image: ', img.nbytes)
    image.show(img)
    image.save(img,filename[:-4]+'_gray.bmp')
    plt.show()
    print('Compressed image: ', compressed)
    image.show(u@np.diag(s)@v)
    image.save(u@np.diag(s)@v,filename[:-4]+'_rmax.bmp')
    plt.show()
    print('Compression ratio: ', ratio,'\n')
    return img
    

def size_svd(img,r, title):
    u,s,v = np.linalg.svd(img)
    plt.semilogy(np.linspace(0,s.shape[0],s.shape[0]), (s), 'k.')
    plt.title('Singular values: '+title+'image', fontsize=18)
    plt.xlabel(r'x', fontsize=12)
    plt.ylabel(r'log($\sigma_x$)', fontsize=12)
    plt.savefig(title+'_sigma.pdf')
    return (u[:,:r].nbytes + s[:r].nbytes + v[:r].nbytes),u[:,:r],s[:r],v[:r]