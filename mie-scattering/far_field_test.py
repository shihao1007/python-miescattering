# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:39:05 2019

This script verifies the accuracy of the far field simulation by comparing the
field outside the sphere with the near field simulation

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt

#%%
# Calculate the sphere scattering coefficients
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

    # return ai * (bi - ci) / (di - ei)
    return (bi - ci) / (di - ei)

def cal_near_field(res, fov, a, n, lambDa, z_slice):

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
#    E_obj = planewave(k, E)
#    Ep = E_obj.evaluate(x, y, z)
#    Ef = Ep[0,...]
    
    # calculate B vector
    B = coeff_b(l, kMag, n, a)
    
    Es = np.sum(scatter_matrix * B, axis = 2)
    Et = Es #+ Ef
    
    mask = rMag < a
    
    Et[mask] = 0
    
    return Et, mask

#%%
def cal_Es_far_field(a, n, k, k_dir, simRes, simFov, working_dis, scale_factor):
# calculate the scattering field through far field simulation
    
    # the maximal order
    l_max = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    l = np.arange(0, l_max+1, 1)
    
    # calculate B coefficient
    B = coeff_b(l, k, n, a)
    
    # construct the evaluate plane
    # halfgrid is the size of a half grid
    halfgrid = np.ceil(fov/2)*(2*padding +1)
    # range of x, y
    gx = np.linspace(-halfgrid, +halfgrid, simRes)
    gy = gx
    [x, y] = np.meshgrid(gx, gy)     
    # make it a plane at z = 0 (plus the working distance) on the Z axis
    z = np.zeros((simRes, simRes,)) + working_dis
    
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
    # calculate k dot r
    kr = kMag * rMag
    
    # calculate the asymptotic form of hankel funtions
    hlkr_asym = np.zeros((kr.shape[0], kr.shape[1], l.shape[0]), dtype = np.complex128)
    for i in l:
        # <<Recovery of absorption spectra from Fourier transform infrared...>>
        hlkr_asym[..., i] = np.exp(1j*(kr-i*math.pi/2))/(1j * kr)
        
        # Robert G Brown, Duke University
        # kr -> infinity
        #hlkr_asym[..., i] = np.exp(1j*(kr-(i+1)*math.pi/2))/kr
        
    # calculate the legendre polynomial
    # get the frequency components
    fx = np.fft.fftfreq(simRes, simFov/simRes)
    fy = fx
    
    # create a meshgrid in the Fourier Domain
    [kx, ky] = np.meshgrid(fx, fy)
    # calculate the sum of kx ky components so we can calculate 
    # cos_theta in the Fourier Domain later
    kxky = kx ** 2 + ky ** 2
    # create a mask where the sum of kx^2 + ky^2 is 
    # bigger than 1 (where kz is not defined)
    mask = kxky > 1
    # mask out the sum
    kxky[mask] = 0
    # calculate cos theta in Fourier domain
    cos_theta = np.sqrt(1 - kxky)
    cos_theta[mask] = 0
    # calculate the Legendre Polynomial term
    pl_cos_theta = sp.special.eval_legendre(l, cos_theta[..., None])
    # mask out the light that is propagating outside of the objective
    pl_cos_theta[mask] = 0
    
    # calculate the prefix alpha term
    alpha = (2*l + 1) * 1j ** l
    # calculate the matrix besides B vector
    scatter_matrix = hlkr_asym * pl_cos_theta * alpha
    # calculate every order of the integration
    Sum = scatter_matrix * B
    # integrate through all the orders to get the farfield in the Fourier Domain
    E_scatter_fft = np.sum(Sum, axis = -1) * scale_factor
    
    # shift the Forier transform of the scatttering field for visualization
    E_scatter_fftshift = np.fft.fftshift(E_scatter_fft)
    
    # convert back to spatial domain
    E_scattering_b4_shift = np.fft.ifft2(E_scatter_fft)
    
    # shift the scattering field in the spacial domain for visualization
    E_scattering = np.fft.fftshift(E_scattering_b4_shift)
    
    return E_scattering, E_scatter_fftshift

#%%
# set the size and resolution of both planes
fov = 32                    # field of view
res = 256                   # resolution
a = 1                       # radius of the spere
lambDa = 1                  # wavelength
n = 1.5            # refractive index
k = 2 * math.pi / lambDa    # wavenumber
padding = 0                 # padding
working_dis = 100000 * (2 * padding + 1)         # working distance
scale_factor = working_dis * 2 * math.pi * (res/fov)            # scale factor of the intensity
# simulation resolution
# in order to do fft and ifft, expand the image use padding
simRes = res*(2*padding + 1)
simFov = fov*(2*padding + 1)
ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave

center = int(simRes/2)
#%%
E_scattering_far, E_scatter_fftshift = cal_Es_far_field(a, n, k, k_dir,
                                                    simRes, simFov,
                                                    working_dis, scale_factor)

E_scattering_near, mask = cal_near_field(simRes, simFov, a, n, lambDa, -a)

E_scattering_far[mask] = 0

#%%
plt.figure()
plt.subplot(231)
plt.imshow(np.real(E_scattering_near), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Near Field, Real')
plt.colorbar()

plt.subplot(232)
plt.imshow(np.imag(E_scattering_near), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Near Field, Imaginary')
plt.colorbar()

plt.subplot(233)
plt.imshow(np.abs(E_scattering_near), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Near Field, Abs')
plt.colorbar()

plt.subplot(234)
plt.imshow(np.real(E_scatter_fftshift), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Far Field, Real')
plt.colorbar()

plt.subplot(235)
plt.imshow(np.imag(E_scatter_fftshift), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Far Field, Imaginary')
plt.colorbar()

plt.subplot(236)
plt.imshow(np.abs(E_scatter_fftshift), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Far Field, Abs')
plt.colorbar()

#%%
#plt.figure()
#plt.subplot(131)
#plt.plot(np.abs(E_scattering_far[center, center:]), label='Far')
#plt.plot(np.abs(E_scattering_near[center, center:]), label='Near')
#plt.title('Magnitude')
#plt.legend()
#
#plt.subplot(132)
#plt.plot(np.real(E_scattering_far[center, center:]), label='Far')
#plt.plot(np.real(E_scattering_near[center, center:]), label='Near')
#plt.title('Real')
#plt.legend()
#
#plt.subplot(133)
#plt.plot(np.imag(E_scattering_far[center, center:]), label='Far')
#plt.plot(np.imag(E_scattering_near[center, center:]), label='Near')
#plt.title('Imaginary')
#plt.legend()
#
#plt.suptitle(n)