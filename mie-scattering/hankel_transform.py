# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:11:37 2019

Discrete Hankel Transform

Editor:
    Shihao Ran
    STIM Laboratory
"""

import scipy as sp
import scipy.special
import numpy as np
import matplotlib.pyplot as plt

# define a discrete function
def discrete(x, alpha=1):
    if np.isscalar(x):
        if x < alpha:
            return 1
        else:
            return 0
    else:
        res = []
        for i in x:
            if i < alpha:
                res.append(1)
            else:
                res.append(0)
        return res

def bessel_div_r(r):
    return sp.special.jv(1, r)/r
 
#%%
        
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
    jv_mX = jv_m/X
    
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
    


#plt.plot(F)
#plt.show()

#%%
fov = 40
res = 128
#padding = 1 
simFov = int(fov*(padding*2 + 1))
simRes = int(res*(padding*2 + 1))
length = int(simRes/2)
x = np.linspace(-simFov/2, simFov/2, simRes)[int(simRes/2):int(simRes/2)+length]


#%%
n = near_line_gt
F, F_term, F_x, F_fit_x = idhf(simFov, simRes, x, far_line)
F_n = (F-np.mean(F))/np.std(F)
n_n = (n-np.mean(n))/np.std(n)

#%%
plt.figure()
#plt.subplot(121)
plt.plot(F_x, np.real(F_n), label='Transformed')
plt.plot(x, np.real(n_n), label='Ground Truth')
plt.ylabel('Intensity')
plt.xlabel('Pixel Index')
plt.legend()
plt.title('Hankel Transform')

#plt.subplot(122)
#plt.plot(F_fit_x, np.real(F_term), marker='.', linestyle='none', ms=4, alpha=0.6, label='Curve Fit')
#plt.plot(x, np.real(far_line), marker='.', ms=2, alpha=0.6, label='Far Field Line')
#plt.xlabel('Half FOV / um')
#plt.title('Interpolation')
#plt.legend()

plt.suptitle('Padding '+str(padding))
plt.show()

#%%
#F_test = idhf(simFov, simRes, discrete)
#
#epsi = 10**(-14)
#x_f = np.linspace(epsi, simFov/2+epsi, int(simRes/2))
#n_test = sp.special.jv(1, x_f)/x_f
#
#plt.figure()
##plt.title('Spatial Domain')
#plt.plot(x_f, discrete(x_f, 1))
#plt.title('Reciprocal Domain')
#plt.xlabel('r')
#plt.ylabel('F')
#
##%%
#plt.figure()
#plt.title('Padding '+str(padding))
#plt.plot(F_test, label='Transformed')
#plt.plot(n_test, label='Ground Truth')
#plt.legend()
#plt.show()
#
##%%
#plt.figure()
#plt.plot(x_f, discrete(x_f, 1))
#plt.title('Staircase Function')