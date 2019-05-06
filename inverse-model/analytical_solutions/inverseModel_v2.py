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
            
def getB(a, n, r, Et, Ei, k, lambDa):
    
    numOrd = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    ordVec = np.arange(0, numOrd+1, 1)
    
    R = Et / Ei - 1
    
    r_mag = np.sqrt(r[:, 0]**2 + r[:, 1]**2 + r[:, 2]**2)
    k_mag = 2*np.pi/lambDa
    
    r_norm = r / r_mag[..., None]
    
    kr = r_mag * k_mag
    hlkr = sphhankel(ordVec, kr, 0)
    
    cosTheta = np.dot(r_norm, k)
    plcos = Legendre(numOrd, cosTheta)
    
    preterm = (2* ordVec + 1) * 1j ** ordVec 
    
    hlr = preterm * hlkr * plcos
    
    B = np.linalg.solve(hlr, R)
    
    return B

def getMultiB(num_cal, numOrd):
    
    B = np.zeros((numOrd+1), dtype = np.complex128)
    for i in range(num_cal):
        
        r_start = np.shape(rVecs)[0]//2 + 1 + i
        r_inv = rVecs[r_start,r_start:r_start+numOrd+1,:]
        
        Et_inv = Et[r_start,r_start:r_start+numOrd+1]
        Ei = 1
        
        B_test = getB(a, n, r_inv, Et_inv, Ei, k, lambDa)

        B += B_test
    
    B /= num_cal
    
    return B


#Et = Et_0
Et = Et_plus_noise

a = 6
k = np.array([0, 0, -1])
n = 1.5 + 0.1j
lambDa = 2*np.pi

numAve = 6
numOrd = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)

B_test = getMultiB(numAve, numOrd)


plotthings(np.squeeze(B0), B_test)