# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 09:32:35 2019

@author: shihao

Inverse model for CHIS
Use the simulated fields to calculate the size a 
and refractive index n of the sphere
Solve a linear system to get the B coeffiect vector
Optimize a non-linear problem to predict the value of a and n
"""

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

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
        ans = np.zeros((x.shape + (len(order))), dtype = np.complex128)
        for i in range(len(order)):
            ans[...,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
        return ans

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
        
    
##for test use
Et = Et_h
Ef = np.ones((50, 50), dtype = np.complex128)
k = np.asarray([0, 0, -1])
z = 17

#assume a number of order
#this value could be optimized with iteractive calculation
#since it is related to the size of the sphere a
num_order = 24

def getBrandom():
    
    #import the fields from the previously calculated results
    #field with the sample present
    E_sample = Et
    #field without the sample present
    E_without_sample = 1
    
    #the ratio between the two fields
    #the right-hand side of the linear system
    R_field = E_sample / E_without_sample - 1
    
    #sample the number of R equal to the number of order
    #get randomized values form R matrix to form a vector as the right-hand side of the linear system
    R_points = np.asarray([R_field[i, np.shape(E_sample)[0]//2] for i in range(np.shape(E_sample)[0]//2)])
    
    #get the 3-D r matrix
    r = np.asarray([rVecs[i, j] for i in range(5) for j in range(5)])
    
    
    #calculate the length of r and k vectors
    r_mag = np.asarray([np.sqrt(r[i,0]**2 + r[i,1] ** 2 + r[i,2] ** 2) for i in range(num_order+1)])
    k_mag = 2*np.pi/8
    
    #compute the scalar product of k and r
    k_r = r_mag * k_mag 
    
    #compose a vector that contains all number of orders
    ordVec = np.arange(0, num_order+1, 1)
    
    #calculate the spherical hankel function of the 1st kind for all the orders
    h_kr = sphhankel(ordVec, k_r, 0)
    
    #normalize k and r vectors for computing the angle between them
    k_norm = k / np.linalg.norm(k)
#    r_norm = r / r_mag
    
    #compute the cos of angle between k and r vectors
    cosTheta = (np.asarray([cos_theta[i, j] for i in range(5) for j in range(5)]))
    
    #compute the Legendre polynomials of all the orders
    Plcos = Legendre(num_order, cosTheta)
    
    #compute the prefix term of H(r), (2l + 1) * i ^ l
    pre_term = (2 * ordVec + 1) * 1j ** ordVec
    
    HlR0 = pre_term * (np.asarray([hlkrcos[i, j] for i in range(np.shape(E_sample)[0]) for j in range(np.shape(E_sample)[0])]))
    HlR = h_kr * Plcos *  pre_term 
    
    B = np.linalg.solve(HlR, R_points)
    
    HlR_inv = np.linalg.inv(HlR)
    
    B = np.dot(R_points, HlR_inv)
    return B


def getBpseudo():
    
    #import the fields from the previously calculated results
    #field with the sample present
    E_sample = Et
    #field without the sample present
    E_without_sample = Ef
    
    #the ratio between the two fields
    #the right-hand side of the linear system
    R_field = E_sample / E_without_sample - 1
    
    D = np.asarray([R_field[i,j] for i in range(np.shape(E_sample)[0]) for j in range(np.shape(E_sample)[0])])[::int(np.size(E_sample)/num_seudo)]
        
    #get the 3-D r matrix
    r = np.asarray([rVecs[i, j] for i in range(np.shape(E_sample)[0]) for j in range(np.shape(E_sample)[0])])[::int(np.size(E_sample)/num_seudo),:]
    
    #calculate the length of r and k vectors
    r_mag = np.asarray([np.sqrt(r[i,0]**2 + r[i,1] ** 2 + r[i,2] ** 2) for i in range(np.shape(r)[0]+1)])
    k_mag = 2*np.pi/8
    
    #compute the scalar product of k and r
    k_r = r_mag * k_mag 
    
    #compose a vector that contains all number of orders
    ordVec = np.arange(0, num_order+1, 1)
    
    #calculate the spherical hankel function of the 1st kind for all the orders
    h_kr = sphhankel(ordVec, k_r, 0)
    
    #normalize k and r vectors for computing the angle between them
#    k_norm = k / np.linalg.norm(k)
#    r_norm = r / r_mag
    
    #compute the cos of angle between k and r vectors
    cosTheta = (np.asarray([cos_theta[i, j] for i in range(np.shape(E_sample)[0]) for j in range(np.shape(E_sample)[0])]))[::int(np.size(E_sample)/num_seudo)]
    
    #compute the Legendre polynomials of all the orders
    Plcos = Legendre(num_order, cosTheta)
    
    #compute the prefix term of H(r), (2l + 1) * i ^ l
    pre_term = (2 * ordVec + 1) * 1j ** ordVec
    
    HlR0 = pre_term * (np.asarray([hlkrcos[i, j] for i in range(np.shape(E_sample)[0]) for j in range(np.shape(E_sample)[0])]))
    HlR =  h_kr * Plcos * pre_term
    
    P = np.matrix(HlR)                          #set array HlR as matrix P
    Ph = P.getH()                               #get hermintian inverse of matrix HlR
    PhP_inverse = np.linalg.inv(np.dot(Ph, P))         #compute the inverse of Ph * P
    PhPPh = np.dot(PhP_inverse, Ph)                    

    B = np.asarray(np.dot(PhPPh, D))
    
    return B


#B = np.zeros((21), dtype = np.complex128)
#for _ in range(100):
#    B += getB()
#d = 110
#B = getBuniform(d)    

B = getBrandom()

hlr0_s = np.asarray([hlr0[i, np.shape(E_sample)[0]//2 + 5] for i in range(np.shape(E_sample)[0]//2)])

#num_seudo = 90000
#B = getBpseudo()

def plotthings(x):
    
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.plot(np.abs(x))
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Magnitude')
    
    plt.subplot(1, 3, 2)
    plt.plot(np.real(x))
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Real')
    
    plt.subplot(1, 3, 3)
    plt.plot(np.imag(x))
    plt.xlabel('Number of Order')
    plt.ylabel('Value')
    plt.title('Imaginary')
    
#    plt.suptitle('B: Pseudo Inverse '+str(num_seudo)+' Samples')
    
def showthings(x):
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(np.abs(h))
    plt.title('Magnitude')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(np.real(x))
    plt.title('Real')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(np.imag(x))
    plt.title('Imaginary')
    plt.colorbar()
    
    plt.suptitle('H Matrix')
    
showthings(hlr)
plotthings(np.squeeze(Bs))
