"""
Create a near field point spread function by the far field simulation
"""

import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt

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


# set the size and resolution of both planes
fov =  32
res = 518
a = 1                       # radius of the sphere
n = 1                       # refractive index
k = 2 * math.pi
Nl = 100                     # number of field orders

# set the numerical aperture
NA_in = 0.0
NA_out = 1.0

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

# calculate the coordinates of k values that make it into the objective
#[idx, idy] = np.nonzero(K < k)

# the scale factor to increase the intensity in spatial domain
intensity_scale = 100

# scale the Fourier transform of the psf by kz
psf_f = np.ones(kz.shape) / kz * intensity_scale
psf_f[mask] = 0

#%%
psf_f_shift = np.fft.fftshift(psf_f)
psf = np.fft.ifft2(psf_f)

psf_shift = np.fft.ifftshift(psf)
#%%
plt.figure()

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