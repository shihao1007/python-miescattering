# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:04:11 2019

this script verifies the fourier transform and hankel transform of a 
normal mie scattering field

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.special

#%%
#E_fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(Et_close)))
#
#E_real = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(E_fft)))
#
#plt.figure()
#plt.set_cmap('RdYlBu')
#plt.subplot(121)
#plt.imshow(np.real(E_real))
#plt.title('Spatial Domain, Real')
#plt.colorbar()
#
#plt.subplot(122)
#plt.imshow(np.real(E_fft))
#plt.title('Fourier Domain, Real')
#plt.colorbar()
#
#length = int(simRes/2)
#center_idx = int(simRes/2)
#E_near = E_real
#near_line_gt = E_near[center_idx, center_idx:center_idx+length]
#far_line = E_fft[center_idx, center_idx:center_idx+length]

def idhf(simFov, simRes, x, y):
    """
    Inverse Discrete Hankel Transform of an 1D array
    param:
        order: order of the bessel function
        func: the function being transformed
    """
    
    X = int(simFov/2)
    n_s = int(simRes/2)
    
    # order of the bessel function
    order = 0
    # root of the bessel function
    jv_root = sp.special.jn_zeros(order, n_s)
    jv_M = jv_root[-1]
    jv_m = jv_root[:-1]
    jv_mX = jv_m/(X* 2*np.pi)
    
#    F_term = np.interp(jv_mX, x, y)
    F_term = y[:-1]
    # inverse DHT
    F = np.zeros(n_s, dtype=np.complex128)
    for k in range(n_s):
        prefix = 2/(X**2)
        
        summation = 0
        for m in range(n_s-1):
            
            Jjj = jv_root[m]*jv_root[k]/jv_M
            numerator = sp.special.jv(order, Jjj)
            denominator = sp.special.jv(order+1, jv_root[m])**2
            
            summation += F_term[m] * numerator / denominator
        
        F[k] = prefix * summation

    return F, F_term, jv_root*X/jv_M, jv_mX

#%%

fov = 10
res = 400
r = np.linspace(-fov/2, fov/2, res)
xx, yy = np.meshgrid(r, r)

im = sp.special.j0(xx**2 + yy**2)



#%%

im_fft = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(im)))


#%%
x = r[int(res/2):]
line_fft = im_fft[int(res/2), int(res/2):]
#plt.figure()
#plt.plot(line_fft)

#%%
F, F_term, tn, un = idhf(fov, res, x, line_fft)

n= im[int(res/2), int(res/2):]
F_n = (F-np.mean(F))/np.std(F)
n_n = (n-np.mean(n))/np.std(n)

#%%

plt.figure()
plt.subplot(221)
plt.imshow(im)
plt.title('Spacial Domain Image')


plt.subplot(222)
plt.imshow(np.real(im_fft))
plt.title('Frequency Domain Real')

plt.subplot(223)
plt.title('Hankel Transform')
plt.plot(x, n_n, label='Ground Truth')
plt.plot(tn, F_n, label='Transformed')
plt.legend()

plt.subplot(224)
plt.title('Interpolation')
plt.plot(x, line_fft, label='Line FFt')
plt.plot(un, F_term, marker='.', linestyle='none', ms=2, label='Interpolation')
plt.legend()

plt.suptitle('sp.special.j0(xx**2 + yy**2)')