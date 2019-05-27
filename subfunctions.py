# -*- coding: utf-8 -*-

import numpy as np

def correlation_fct(img):
    m = np.shape(img)[0]
    n = np.shape(img)[1]
    
    c11 = np.fft.fft2(img)/( m*n )
    
    c11 = np.multiply( np.conj(c11) , c11 )
    c11 = np.fft.ifft2(c11)
    
    c11 = m*n*c11.real
    
    return c11
    
def NN_input(B,img,c11):    
    f_b = np.mean(img)    
    xi = np.matmul(np.transpose(B), (c11-f_b**2)/f_b )

    x=np.vstack( (f_b,xi) )    
    
    return x

