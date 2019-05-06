# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 14:29:43 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""

import time
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

def exp(x):
    if np.isscalar(x):
        return 1/x
    else:
        res = []
        for i in x:
            res.append(1/i)
        return res
    
def idhf(simFov, simRes, func):
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
    jv_mX = jv_root[:-1]/X

    # inverse DHT
    F = np.zeros(n_s, dtype=np.complex128)
    
    for k in range(n_s):
        F_fit = []
        prefix = 2/(X**2)       
        summation = 0
        for m in range(n_s-1):
            
            x_f = jv_root[m]/X
            F_term = func(x_f)
            
            Jjj = jv_root[m]*jv_root[k]/jv_M
            numerator = sp.special.jv(order, Jjj)
            denominator = sp.special.jv(order+1, jv_root[m])**2
            
            summation += F_term * numerator / denominator
            F_fit.append(F_term)
            
        F[k] = prefix * summation
        

    return F, F_fit, jv_root*X/jv_M, jv_mX

#%%
fov = 16
res = 128
padding = 4
simFov = int(fov*(padding*2 + 1))
simRes = int(res*(padding*2 + 1))
x = np.linspace(-simFov/2, simFov/2, simRes)[int(simRes/2):]
start = time.time()
#F_test, F_fit = idhf(simFov, simRes, discrete)
end = time.time()
epsi = 1
x_f = np.linspace(epsi, simFov/2+epsi, int(simRes/2))
#n_test = sp.special.jv(1, x_f)/x_f
#

F_test1, F_fit1, F_x, F_fit_x = idhf(simFov, simRes, exp)


print(end-start)
n_test1 = 1/x_f
#%%
plt.figure()
plt.subplot(121)
plt.title('Exponential Function')
plt.plot(F_x, F_test1, label='Transformed')
plt.plot(x_f, n_test1, label='Ground Truth')
plt.legend()

plt.subplot(122)
plt.title('Exponential Root Samples')
plt.plot(F_fit_x, F_fit1, label='With Roots')
plt.plot(x_f, exp(x_f), label='Ground Truth')
plt.legend()