# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 09:41:23 2019

create a plot of the error trend of different parameters into the propagation function

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

#%%
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
    # make it a plane at z = z_slice
    # make sure the z plane is on the other side of the sphere along the propagation direction
    if z_slice != 0:
        if k_dir[2] * z_slice < 0:
                z_slice *= -1
        elif k_dir[2] * z_slice == 0:
            print('Please specify k vector direction at z axis')
            
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

def get_error(res, fov, a, n, lambDa, z0_slice, z_distance):
    
    E0, mask = cal_field(res, fov, a, n, lambDa, z0_slice)
    Ez, temp = cal_field(res, fov, a, n, lambDa, z0_slice + z_distance)

    # propagate the field to the same plane as E0
    E_prop, phase = propagate_field(Ez, fov, 2 * np.pi / lambDa, z_distance)
    E_prop[mask] = 0
    
    # calculate the average pixel error after propagation
    E_error = E_prop - E0
    Error_real = np.average(np.real(E_error))
    Error_imag = np.average(np.imag(E_error))
    Error_abs = np.average(np.abs(E_error))
    
    return Error_real, Error_imag, Error_abs
    
#%%
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
NA_out = 0
# wave length
lambDa = 1
# padding size
padding = 0
# sphere radius
a = 0.8
# test sample number
num_test = 30
# fixed plane position
z0_slice = 0

# test parameters
# propagation distance
z_distance_min = 1
z_distance_max = 3

z_distance_list = np.linspace(z_distance_min, z_distance_max, num_test)

# refractive index
nr_min = 1.0
nr_max = 2.0

nr_list = np.linspace(nr_min, nr_max, num_test)

# attenuation coefficient
ni_min = 0.0
ni_max = 0.3

ni_list = np.linspace(ni_min, ni_max, num_test)

#%%
# initialize test results
# test along propagation distance
error_distance_r = np.zeros(num_test)
error_distance_i = np.zeros(num_test)
error_distance_a = np.zeros(num_test)

n_test = 1.05

for d_idex, distance in enumerate(z_distance_list):
    
    [error_distance_r[d_idex],
     error_distance_i[d_idex],
     error_distance_a[d_idex]] = get_error(res, fov, a, n_test, lambDa, z0_slice, distance)

#%%
# plot the error of distance
plt.figure()
plt.plot(z_distance_list, error_distance_r, label = 'Real')
plt.plot(z_distance_list, error_distance_i, label = 'Imaginary')
plt.plot(z_distance_list, error_distance_a, label = 'Abs')
plt.xlabel('Propagation Distance')
plt.ylabel('Averaged Error')
plt.title('E0 at ' + str(z0_slice) + ', n = ' + str(n_test) + ', a = ' + str(a))
plt.legend()


#%%
# initialize test results
# test along propagation distance
error_nr_r = np.zeros(num_test)
error_nr_i = np.zeros(num_test)
error_nr_a = np.zeros(num_test)

distance_test = 1.5

for nr_idex, nr in enumerate(nr_list):
    
    [error_nr_r[nr_idex],
     error_nr_i[nr_idex],
     error_nr_a[nr_idex]] = get_error(res, fov, a, nr, lambDa, z0_slice, distance_test)

#%%
# plot the error of distance
plt.figure()
plt.plot(nr_list, error_nr_r, label = 'Real')
plt.plot(nr_list, error_nr_r, label = 'Imaginary')
plt.plot(nr_list, error_nr_r, label = 'Abs')
plt.xlabel('Refractive Index')
plt.ylabel('Averaged Error')
plt.title('E0 at ' + str(z0_slice) + ', distance = ' + str(distance_test) + ', a = ' + str(a))
plt.legend()

#%%
# initialize test results
# test along propagation distance
error_ni_r = np.zeros(num_test)
error_ni_i = np.zeros(num_test)
error_ni_a = np.zeros(num_test)

distance_test = 1.5
nr_test = 1.05

for ni_idex, ni in enumerate(ni_list):
    
    [error_ni_r[ni_idex],
     error_ni_i[ni_idex],
     error_ni_a[ni_idex]] = get_error(res, fov, a, nr_test + ni * 1j, lambDa, z0_slice, distance_test)

#%%
# plot the error of distance
plt.figure()
plt.plot(ni_list, error_ni_r, label = 'Real')
plt.plot(ni_list, error_ni_r, label = 'Imaginary')
plt.plot(ni_list, error_ni_r, label = 'Abs')
plt.xlabel('Attenuation Coefficient')
plt.ylabel('Averaged Error')
plt.title('E0 at ' + str(z0_slice) + ', distance = ' + str(distance_test) + ', a = ' + str(a) + ', n = ' + str(nr_test))
plt.legend()