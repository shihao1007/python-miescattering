# -*- coding: utf-8 -*-
"""
Created on Mon May  6 12:12:25 2019

Create a near field point spread function
Using the asymptotic form (far field simulation)

Editor:
    Shihao Ran
    STIM Laboratory
"""

# import packages
import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt

#%%
# set the parameters for the simulation
# field of view
fov =  32
# resolution
res = 518
# radius of the sphere
a = 1
# refractive index of the sphere
n = 1
# wavenumber of the incident field
k = 2 * math.pi
# number of field orders
Nl = 100

# set the numerical aperture
NA_in = 0.0
NA_out = 1.0

#%%
# calculate the coordinates in the Fourier domain
kx = np.fft.fftfreq(res, fov/res) * 2 * math.pi
fx = np.fft.fftshift(kx)/ (2 * math.pi)
ky = kx

# create a grid to store the simulation
[KX, KY] = np.meshgrid(kx, ky)

# calculate kx^2 + ky^2
kxky = KX ** 2 + KY ** 2

# mask out the plane waves go into the objective
mask_out = k**2 * NA_out**2 <= kxky
mask_in = kxky < k**2 * NA_in ** 2
mask = mask_out + mask_in

# apply mask and calculate kz
kxky[mask] = 0
kz = np.sqrt(k**2 - kxky)
kz[mask] = 0

# the scale factor to increase the intensity in spatial domain
intensity_scale = 100

# scale the Fourier transform of the psf by kz
psf_f = np.ones(kz.shape) / kz * intensity_scale
psf_f[mask] = 0

#%%
# convert the field from the fourier domain into spatial domain
psf_f_shift = np.fft.fftshift(psf_f)
psf = np.fft.ifft2(psf_f)

psf_shift = np.fft.ifftshift(psf)
#%%
# plot the image
plt.figure()
plt.set_cmap('RdYlBu')

plt.subplot(141)
plt.imshow(np.real(psf_shift), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title('Real')
plt.colorbar()

plt.subplot(142)
plt.imshow(np.imag(psf_shift), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title('Real')
plt.colorbar()

plt.subplot(143)
plt.imshow(np.abs(psf_shift), extent = [-fov/2, fov/2, -fov/2, fov/2])
plt.title('Real')
plt.colorbar()

plt.subplot(144)
plt.imshow(psf_f_shift, extent = [fx[0], fx[-1], fx[0], fx[-1]])
plt.title('PSF Fourier Domain')
plt.colorbar()