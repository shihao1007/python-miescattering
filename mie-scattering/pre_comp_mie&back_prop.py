# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:24:56 2019

Calculate forward Mie-scattering and apply back propagation

Editor:
    Shihao Ran
    STIM Laboratory
"""

# numpy for most of the data saving and cumputation
import numpy as np
# matplotlib for ploting the images
from matplotlib import pyplot as plt
# pyquaternion for ratating the vectors
from pyquaternion import Quaternion
# scipy for input/output files and special computing
import scipy as sp
from scipy import special
# math for calculations
import math
# import animation for plot animations
from matplotlib import animation as animation
# import sys for printing progress
import sys
# import random for MC sampling
import random

def propagate_field(E, fov, k, z):
    
    # calculate the fourier frequencies
    fx = np.fft.fftfreq(E.shape[0], fov/E.shape[0]) * k
    fy = np.fft.fftfreq(E.shape[1], fov/E.shape[1]) * k
    
    # calculate kz
    kx, ky = np.meshgrid(fx, fy)
    kxky = kx ** 2 + ky ** 2
    mask = kxky > k**2
    kxky[mask] = 0
    kz = np.sqrt(k**2 - kxky)
    kz[mask] = 0
    
#    plt.figure()
#    plt.imshow(kz)
#    plt.colorbar()
    
    # calculate phase term
    phase = np.exp(1j * kz * z)
    
    # apply phase shift in Fourier Domain
    E_FFT = np.fft.fft2(E)
    E_prop_FFT = E_FFT * phase
    
    # inverse Fourier Transform
    E_prop = np.fft.ifft2(E_prop_FFT)
    
    return E_prop, phase

class planewave():
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    def __init__ (self, k, E):
        
        #self.phi = phi
        self.k = k/np.linalg.norm(k)                      
        self.E = E
        
        #force E and k to be orthogonal
        if ( np.linalg.norm(k) > 1e-15 and np.linalg.norm(E) >1e-15):
            s = np.cross(k, E)              #compute an orthogonal side vector
            s = s / np.linalg.norm(s)       #normalize it
            Edir = np.cross(s, k)              #compute new E vector which is orthogonal
            self.k = k
            self.E = Edir / np.linalg.norm(Edir) * np.linalg.norm(E)
    
    def __str__(self):
        return str(self.k) + "\n" + str(self.E)     #for verify field vectors use print command

    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z     #phase term k*r
        ex = np.exp(1j * k_dot_r)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        Ef = self.E.reshape((3, 1, 1)) * ex
        return Ef

def coeff_b(l, k, n, a):
    jka = sp.special.spherical_jn(l, k * a)
    jka_p = sp.special.spherical_jn(l, k * a, derivative=True)
    jkna = sp.special.spherical_jn(l, k * n * a)
    jkna_p = sp.special.spherical_jn(l, k * n * a, derivative=True)

    yka = sp.special.spherical_yn(l, k * a)
    yka_p = sp.special.spherical_yn(l, k * a, derivative=True)

    hka = jka + yka * 1j
    hka_p = jka_p + yka_p * 1j

    bi = jka * jkna_p * n
    ci = jkna * jka_p
    di = jkna * hka_p
    ei = hka * jkna_p * n

    return (bi - ci) / (di - ei)

def cal_field(res, fov, a, n, lambDa, z_slice):

    # the maximal order
    l_max = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    l = np.arange(0, l_max+1, 1)
    
    # construct the evaluate plane
    # simulation resolution
    # in order to do fft and ifft, expand the image use padding
    simRes = res*(2*padding + 1)
    # initialize a plane to evaluate the field
    # halfgrid is the size of a half grid
    halfgrid = np.ceil(fov/2)*(2*padding +1)
    # range of x, y
    gx = np.linspace(-halfgrid, +halfgrid, simRes)
    gy = gx
    [x, y] = np.meshgrid(gx, gy)     
    # make it a plane at z = 0 on the Z axis
    z = np.zeros((simRes, simRes,)) + z_slice
    
    # initialize r vectors in the space
    rVecs = np.zeros((simRes, simRes, 3))
    # make x, y, z components
    rVecs[:,:,0] = x
    rVecs[:,:,1] = y
    rVecs[:,:,2] = z
    # compute the rvector relative to the sphere
    rVecs_ps = rVecs - ps
    
    # calculate the distance matrix
    rMag = np.sqrt(np.sum(rVecs_ps ** 2, 2))
    kMag = 2 * np.pi / lambDa
    k = np.asarray(k_dir) * kMag
    
    rNorm = rVecs_ps / rMag[...,None]
    
    # preconpute scatter matrix = hlkr * plcos_theta
    
    kr = kMag * rMag
    cos_theta = np.dot(rNorm, k_dir)
    
    # calculate hlkr and plcos
    scatter_matrix = np.zeros((simRes, simRes, l_max+1), dtype = np.complex128)
    
    alpha = (2 * l + 1) * (1j ** l) 
    jkr = sp.special.spherical_jn(l, kr[..., None])
    ykr = sp.special.spherical_yn(l, kr[..., None])
    hlkr = jkr + ykr * 1j
    plcos = sp.special.eval_legendre(l, cos_theta[..., None])
    scatter_matrix = hlkr * plcos * alpha
    
    # pre compute Ef, incident field at z-max
    E_obj = planewave(k, E)
    Ep = E_obj.evaluate(x, y, z)
    Ef = Ep[0,...]
    
    # calculate B vector
    B = coeff_b(l, kMag, n, a)
    
    Es = np.sum(scatter_matrix * B, axis = 2)
    Et = Es + Ef
    
    mask = rMag < a
    
    Et[mask] = 0
    
    return Et, mask

# specify parameters for the forward model
# propagation direction vector k
k_dir = [0, 0, -1]
E = [1, 0, 0]
# position of the sphere
ps = [0, 0, 0]
# resolution of the cropped image
res = 128
# field of view
fov = 16
# in and out numerical aperture of the bandpass optics
NA_in = 0
NA_out = 1
# wave length
lambDa = 1
# padding size
padding = 0


z0_slice = -0
z_distance = -1.5
# set ranges of the features of the data set
# refractive index
n = 1.25
# sphere radius
a = 1



E0, mask = cal_field(res, fov, a, n, lambDa, z0_slice)
Ez, temp = cal_field(res, fov, a, n, lambDa, z0_slice + z_distance)


#%%
# plot two fields at different planes
#plt.figure()
#plt.subplot(231)
#plt.imshow(np.real(E0))
#plt.colorbar()
#plt.title('E0, Real')
#plt.subplot(232)
#plt.imshow(np.imag(E0))
#plt.colorbar()
#plt.title('E0, Imaginary')
#plt.subplot(233)
#plt.imshow(np.abs(E0))
#plt.colorbar()
#plt.title('E0, Abs')
#plt.subplot(234)
#plt.imshow(np.real(Ez))
#plt.colorbar()
#plt.title('Ez, Real')
#plt.subplot(235)
#plt.imshow(np.imag(Ez))
#plt.colorbar()
#plt.title('Ez, Imaginary')
#plt.subplot(236)
#plt.imshow(np.abs(Ez))
#plt.colorbar()
#plt.title('Ez, Abs')

#%%
# propagate the field to the same plane as E0
E_prop, phase = propagate_field(Ez, fov, 2 * np.pi / lambDa, z_distance)
E_prop_FFT = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(E_prop)))
#E_prop[mask] = 0

#%%
# plot the fields after propagation and their difference map
#plt.figure()
#plt.subplot(331)
#plt.imshow(np.real(E0))
#plt.colorbar()
#plt.title('E0, Real')
#plt.subplot(332)
#plt.imshow(np.imag(E0))
#plt.colorbar()
#plt.title('E0, Imaginary')
#plt.subplot(333)
#plt.imshow(np.abs(E0))
#plt.colorbar()
#plt.title('E0, Abs')
#
#plt.subplot(334)
#plt.imshow(np.real(E_prop))
#plt.colorbar()
#plt.title('E_prop, Real')
#plt.subplot(335)
#plt.imshow(np.imag(E_prop))
#plt.colorbar()
#plt.title('E_prop, Imaginary')
#plt.subplot(336)
#plt.imshow(np.abs(E_prop))
#plt.colorbar()
#plt.title('E_prop, Abs')
#
#plt.subplot(337)
#plt.imshow(np.real(E_prop) - np.real(E0))
#plt.colorbar()
#plt.title('Difference, Real')
#plt.subplot(338)
#plt.imshow(np.imag(E_prop) - np.imag(E0))
#plt.colorbar()
#plt.title('Difference, Imaginary')
#plt.subplot(339)
#plt.imshow(np.abs(E_prop) - np.abs(E0))
#plt.colorbar()
#plt.title('Difference, Abs')
#
#plt.suptitle('E0 at ' + str(z0_slice) + ', distance = ' + str(z_distance) + ' n = ' + str(n))

#%%
# calculate the average pixel error after propagation
E_error = E_prop - E0
Error = np.average(E_error)

#%%
# plot the fields after propagation and their fft
plt.figure()
plt.subplot(231)
plt.imshow(np.real(E_prop_FFT))
plt.colorbar()
plt.title('E_prop_FFT, Real')
plt.subplot(232)
plt.imshow(np.imag(E_prop_FFT))
plt.colorbar()
plt.title('E_prop_FFT, Imaginary')
plt.subplot(233)
plt.imshow(np.abs(E_prop_FFT))
plt.colorbar()
plt.title('E_prop_FFT, Abs')

plt.subplot(234)
plt.imshow(np.real(E_prop))
plt.colorbar()
plt.title('E_prop, Real')
plt.subplot(235)
plt.imshow(np.imag(E_prop))
plt.colorbar()
plt.title('E_prop, Imaginary')
plt.subplot(236)
plt.imshow(np.abs(E_prop))
plt.colorbar()
plt.title('E_prop, Abs')


plt.suptitle('E0 at ' + str(z0_slice) + ', distance = ' + str(z_distance) + ' n = ' + str(n))