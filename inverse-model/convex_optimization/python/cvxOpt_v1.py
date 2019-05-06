# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 16:31:28 2019

@author: shihao
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 09:37:26 2019

@author: shihao

convex optimization using CVXpy

Version 2 using the whole image
"""

import numpy as np
import cvxpy as cp
from scipy import special as sp
from matplotlib import pyplot as plt
import math
import scipy.io as sio


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
                return np.sqrt(np.pi / (2*x)) * sp.jv(order + 0.5 + mode, x)
            
            elif np.asarray(x).ndim == 1:
                ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
                for i in range(len(order)):
                    ans[:,i] = np.sqrt(np.pi / (2*x)) * sp.jv(i + 0.5 + mode, x)
                return ans
            
            else:
                ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
                for i in range(len(order)):
                    ans[...,i] = np.sqrt(np.pi / (2*x)) * sp.jv(i + 0.5 + mode, x)
                return ans
            
            
            
def sphhankel(order, x, mode):
#general form of calculating spherical hankel functions of the first kind at x
    
    if np.isscalar(x):
        return np.sqrt(np.pi / (2*x)) * (sp.jv(order + 0.5 + mode, x) + 1j * sp.yv(order + 0.5 + mode, x))
#
        
    elif np.asarray(x).ndim == 1:
        ans = np.zeros((len(x), len(order)), dtype = np.complex128)
        for i in range(len(order)):
            ans[:,i] = np.sqrt(np.pi / (2*x)) * (sp.jv(i + 0.5 + mode, x) + 1j * sp.yv(i + 0.5 + mode, x))
        return ans
    else:
        ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
        for i in range(len(order)):
            ans[...,i] = np.sqrt(np.pi / (2*x)) * (sp.jv(i + 0.5 + mode, x) + 1j * sp.yv(i + 0.5 + mode, x))
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

a = 10
k = np.array([0, 0, -1])
n = 1.2 + 0.1j
lambDa = 2*np.pi
num_sample = 16384
step = 128*128//num_sample

numOrd = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
ordVec = np.arange(0, numOrd+1, 1)

r_inv = np.reshape(rVecs, (128*128, 3))

r = r_inv[::step,:]

Et_inv = np.reshape(Et_0, (128*128))

Et = Et_inv[::step]
Ei = 1

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

B0 = np.squeeze(B0)

A = hlr
b = R


x = cp.Variable(shape = (44, 1))



x_r = np.real(B0)
x_i = np.imag(B0)

x_comb = np.concatenate((x_r, x_i), axis = 0)

B = np.real(A)
C = np.imag(A)

u = np.real(b)
v = np.imag(b)

A_up = np.concatenate((B, -C), axis = 1)
A_bt = np.concatenate((C, B), axis = 1)
A_comb = np.concatenate((A_up, A_bt), axis = 0)

b_comb = np.concatenate((u, v), axis = 0)



#A_comb_m = np.matrix(A_comb)
#B0_comb_m = np.matrix.transpose(np.matrix(B0_comb))

#gamma = 0.00001
#theta = 1
#    
#Ax = np.matrix(A_comb) * x
#
#cost = cp.sum_squares(Ax - b_comb) + gamma*cp.norm(x, 1)
#
#obj = cp.Minimize(cost)
#
#constr = [x >= -0.8,x <= 0.2]
#
#prob = cp.Problem(obj, constr)
#
#opt_val = prob.solve()
#
#solution = x.value
    



#plt.figure()
#plt.subplot(211)
#plt.plot(np.real(np.squeeze(B0)), label = 'Ground Truth')
#plt.plot(solution[:22], label = 'Solution')
#plt.title('Real')
#plt.legend()
#
#plt.subplot(212)
#plt.plot(np.imag(np.squeeze(B0)), label = 'Ground Truth')
#plt.plot(solution[22:], label = 'Solution')
#plt.title('Imag')
#plt.legend()
#
#print(prob.status)
#print("Optimal value", opt_val)


sio.savemat(r'D:\irimages\irholography\CVX\A.mat', {'A':A_comb})
sio.savemat(r'D:\irimages\irholography\CVX\b.mat', {'b':b_comb})
sio.savemat(r'D:\irimages\irholography\CVX\x0.mat', {'x0':x_comb})
