# -*- coding: utf-8 -*-

import numpy as np

def correlation_fct(A):
    m = np.shape(A)[0]
    n = np.shape(A)[1]
    
    c1 = np.fft.fft2(A)/( m*n )
    
    c1 = np.multiply( np.conj(c1) , c1 )
    c1 = np.fft.ifft2(c1)
    
    c11 = m*n*c1.real
    
    return c11
    
def NN_input(B,h,img,c11):    
    f_b = np.mean(img)    
    xi = np.matmul(np.transpose(B[:,0:h]),(c11-f_b**2) )

    x=np.vstack( (f_b,xi) )    
    
    return x

