# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 09:52:37 2019

@author: shihao

Inverse model version 3
Induce exponentials to drag down the precision in high order hankel function
To decrease the condition number of the H matrix
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 12:09:19 2019

@author: shihao

CHIS inverse model
"""

import numpy as np
import scipy as sp
import math
from matplotlib import pyplot as plt

#
#
#
#spherical bessel functions and hankel functions
#
#
#

def sphbesselj(order, x, mode):
    #calculate the spherical bessel function of the 1st kind with order specified
        #order: the order to be calculated
        #x: the variable to be calculated
        #mode: 1 stands for prime, -1 stands for derivative, 0 stands for nothing
            if np.isscalar(x):
                return np.sqrt(np.pi / (2*x)) * sp.special.jv(order + 0.5 + mode, x)
            
            elif np.asarray(x).ndim == 1:
                ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
                for i in range(len(order)):
                    ans[:,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
                return ans
            
            else:
                ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
                for i in range(len(order)):
                    ans[...,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
                return ans
            
            
            
def sphhankel(order, x, mode):
#general form of calculating spherical hankel functions of the first kind at x
    
    if np.isscalar(x):
        return np.sqrt(np.pi / (2*x)) * (sp.special.jv(order + 0.5 + mode, x) + 1j * sp.special.yv(order + 0.5 + mode, x))
#
        
    elif np.asarray(x).ndim == 1:
        ans = np.zeros((len(x), len(order)), dtype = np.complex128)
        for i in range(len(order)):
            ans[:,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
        return ans
    else:
        ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
        for i in range(len(order)):
            ans[...,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
        return ans
    

#derivative of the spherical bessel function of the first kind
def derivSphBes(order, x):
    js_n = np.zeros(order.shape, dtype = np.complex128)
    js_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
    js_n_p_1 = np.zeros(order.shape, dtype = np.complex128)
    
    js_n = sphbesselj(order, x, 0)
    js_n_m_1 = sphbesselj(order, x, -1)
    js_n_p_1 = sphbesselj(order, x, 1)
    
    j_p = 1/2 * (js_n_m_1 - (js_n + x * js_n_p_1) / x)
    return j_p

#derivative of the spherical hankel function of the first kind
def derivSphHan(order, x):
    sh_n = np.zeros(order.shape, dtype = np.complex128)
    sh_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
    sh_n_p_1 = np.zeros(order.shape, dtype = np.complex128)

    sh_n = sphhankel(order, x, 0)
    sh_n_m_1 = sphhankel(order, x, -1)
    sh_n_p_1 = sphhankel(order, x, 1)
    
    h_p = 1/2 * (sh_n_m_1 - (sh_n + x * sh_n_p_1) / x)
    return h_p

def Legendre(order, x):
    #calcula order l legendre polynomial
            #order: total order of the polynomial
            #x: array or vector or scalar for the polynomial
            #return an array or vector with all the orders calculated
            
        if np.isscalar(x):
        #if x is just a scalar value
        
            P = np.zeros((order+1, 1))
            P[0] = 1
            if order == 0:
                return P
            P[1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[j+1] = ((2*j+1)/(j+1)) *x *(P[j]) - ((j)/(j+1))*(P[j-1])
            return P
        
        elif np.asarray(x).ndim == 1:
        #if x is a vector
            P = np.zeros((len(x), order+1))
            P[:,0] = 1
            if order == 0:
                return P
            P[:, 1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[:,j+1] = ((2*j+1)/(j+1)) *x *(P[:, j]) - ((j)/(j+1))*(P[:, j-1])
            return P
        
        else:
        #if x is an array
            P = np.zeros((x.shape + (order+1,)))
            P[..., 0] = 1
            if order == 0:
                return P
            P[..., 1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[..., j+1] = ((2*j+1)/(j+1)) *x *(P[..., j]) - ((j)/(j+1))*(P[..., j-1])
            return P

#
#
#
#end of pre-defined functions
#
#
#
            
def plotthings(x1, x2):
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.abs(x1), label = 'B0')
    plt.plot(np.abs(x2), label = 'B_test')
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Magnitude')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(np.real(x1), label = 'B0')
    plt.plot(np.real(x2), label = 'B_test')
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Real')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(np.imag(x1), label = 'B0')
    plt.plot(np.imag(x2), label = 'B_test')
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Imaginary')
    plt.legend()
    
    plt.suptitle('Sphere size ' + str(a))

a = 6
k = np.array([0, 0, -1])
n = 1.5 + 0.1j
lambDa = 2*np.pi

#Et = Et_0
Et = Et_plus_noise
Ei = 1
    
numOrd = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
ordVec = np.arange(0, numOrd+1, 1)

r_start = np.shape(rVecs)[0]//2 + 1
r_inv = rVecs[r_start,r_start:r_start+numOrd+3,:]

r = r_inv

Et_inv = Et[r_start,r_start:r_start+numOrd+3]
Ei = 1

R = Et_inv / Ei - 1

r_mag = np.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
k_mag = 2*np.pi/lambDa

r_norm = r / r_mag[..., None]

kr = r_mag * k_mag

kr_t = np.linspace(1, 5, 30)

bessel2 = sp.special.yv(ordVec, 3)

hankel_t = sphhankel(ordVec, kr_t, 0)

hlkr = sphhankel(ordVec, kr, 0)
hlkr_p = derivSphHan(ordVec, kr)

hlkr_py = np.zeros(((numOrd+1, numOrd+1)), dtype = np.complex128)
for i in ordVec:
    hlkr_py[i] = sp.special.hankel1(ordVec[i], kr)

cosTheta = np.dot(r_norm, k)
plcos = Legendre(numOrd, cosTheta)

preterm = (2* ordVec + 1) * 1j ** ordVec 

hlr = preterm * hlkr * plcos

B = np.linalg.solve(hlr, R)

cond = np.zeros((400))
bias = np.linspace(4.65, 4.7, 400)
for i in range(400):

    comp = 1/(sp.special.gamma(ordVec+1) * (2/bias[i])**ordVec) * 1j

    comp_hlr = comp * hlr

    cond[i] = (np.linalg.cond(comp_hlr))

plt.figure()
plt.plot(bias, cond)
plt.xlabel(r'$ \alpha $')
plt.ylabel('Condition Number of H Matrix')

comp = 1/(sp.special.gamma(ordVec+1) * (2/4.67988)**ordVec) * 1j
comp_hlkr = comp * hlr
print(np.linalg.cond(comp_hlkr))

B_comp = B * np.exp(0.659148*ordVec) * -1j

B_inv_comp = np.linalg.solve(comp_hlkr, R)

plt.figure()
plt.subplot(1,3,1)
plt.plot(np.abs(comp))
plt.title('Magnitude')

plt.subplot(1,3,2)
plt.plot(np.real(comp))
plt.title('Real')

plt.subplot(1,3,3)
plt.plot(np.imag(comp))
plt.title('Imaginary')


plt.figure()
plt.subplot(1,3,1)
for i in range(6):
    plt.plot((np.abs(hankel_t[i])))
plt.title('Magnitude')

plt.subplot(1,3,2)
for i in range(6):
    plt.plot((np.real(hankel_t[i])))
plt.title('Real')

plt.subplot(1,3,3)
for i in range(6):
    plt.plot((np.imag(hankel_t[i])))
plt.title('Imaginary')



plt.figure()
plt.subplot(1,3,1)
for i in range(numOrd+1):
    plt.plot((np.abs(hlr[i])))
plt.title('Magnitude')

plt.subplot(1,3,2)
for i in range(numOrd+1):
    plt.plot((np.real(hlr[i])))
plt.title('Real')

plt.subplot(1,3,3)
for i in range(numOrd+1):
    plt.plot((np.imag(hlr[i])))
plt.title('Imaginary')



hlr_cut = hlr[4:, :15]

comp_cut = (1) / hlr_cut

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(np.abs(comp_hlkr))
plt.title('Magnitude')
plt.colorbar()

plt.subplot(1, 3, 2)
plt.imshow(np.real(comp_hlkr))
plt.title('Real')
plt.colorbar()

plt.subplot(1, 3, 3)
plt.imshow(np.imag(comp_hlkr))
plt.title('Imaginary')
plt.colorbar()

plt.suptitle('Comp Matrix')


plotthings(B_comp, B_inv_comp)
