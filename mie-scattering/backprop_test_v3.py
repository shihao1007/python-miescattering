# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 17:20:16 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np

def propagate_field(E, fov, k, z):
    
    # calculate the fourier frequencies
    fx = np.fft.fftfreq(E.shape[0], fov/E.shape[0]) * 2 * np.pi
    fy = np.fft.fftfreq(E.shape[1], fov/E.shape[1]) * 2 * np.pi
    
    # calculate kz
    kx, ky = np.meshgrid(fx, fy)
    kxky = kx ** 2 + ky ** 2
    mask = kxky > k**2
    kxky[mask] = 0
    kz = np.sqrt(k**2 - kxky)
    kz[mask] = 0
    
    # calculate phase term
    phase = np.exp(1j * kz * z)
    
    # apply phase shift in Fourier Domain
    E_FFT = np.fft.fft2(E)
    E_prop_FFT = E_FFT * phase
    
    # inverse Fourier Transform
    E_prop = np.fft.ifft2(E_prop_FFT)
    
    return E_prop

E_prop = propagate_field(Et_distance, fov, 1, -0.5)

plt.figure()
plt.subplot(231)
plt.imshow(np.real(E_prop))
plt.colorbar()
plt.title('Real propagated')
plt.subplot(232)
plt.imshow(np.imag(E_prop))
plt.colorbar()
plt.title('Imag propagated')
plt.subplot(233)
plt.imshow(np.abs(E_prop))
plt.colorbar()
plt.title('Magnitude propagated')

plt.subplot(234)
plt.imshow(np.real(Et_close))
plt.colorbar()
plt.title('Real E close')
plt.subplot(235)
plt.imshow(np.imag(Et_close))
plt.colorbar()
plt.title('Imag E close')
plt.subplot(236)
plt.imshow(np.abs(Et_close))
plt.colorbar()
plt.title('Magnitude E close')