# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 11:00:16 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""


import scipy as sp
from scipy import fftpack
import numpy as np
import matplotlib.pyplot as plt

#compute the coordinates grid in Fourier domain for the calculation of
#corresponding phase shift value at each pixel
#return the frequency components at z axis in Fourier domain
def cal_kz(fov, simRes):
    #the coordinates in Fourier domain is constructed from the coordinates in
    #spatial domain, specifically,
    #1. Get the pixel size in spatial domain, P_size = FOV / Image_Size
    #2. Fourier domain size, F_size = 1 / P_size
    #3. Make a grid with [-F_size / 2, F_size / 2, same resolution]
    #4. Pixel size in Fourier domain will be 1 / Image_size
    
    #make grid in Fourier domain
    
    kfreq = sp.fftpack.fftfreq(simRes, fov/simRes)*2
    
#    x = np.linspace(-simRes/(fov * 2 * 2 * np.pi), simRes/(fov * 2 * 2 * np.pi), simRes)
    x = kfreq
    xx, yy = np.meshgrid(x, x)
    
    #allocate the frequency components in x and y axis
    k_xy = np.zeros((simRes, simRes, 2))
    k_xy[..., 0], k_xy[..., 1] = xx, yy
    
    #compute the distance of x, y components in Fourier domain
    k_para_square = k_xy[...,0]**2 + k_xy[...,1]**2
    
    #initialize a z-axis frequency components
    k_z = np.zeros(xx.shape)
    
    #compute kz at each pixel
    for i in range(len(k_para_square)):
        for j in range(len(k_para_square)):
            if k_para_square[i, j] < 1:
                k_z[i, j] = np.sqrt(1 - k_para_square[i, j])
    
    #return it
    return k_z, kfreq

fov = 200
res = 128
padding = 1
simRes = (2*padding+1)*res

kz, kfreq = cal_kz(fov, simRes)
kz = np.fft.fftshift(kz)
kfreq = np.fft.fftshift(kfreq)

#lambDa = 2 * np.pi
#
#knorm = k / np.linalg.norm(k)
#
#kx = knorm[0] * 2*np.pi*128/30
#
#ky = knorm[1] * 2*np.pi * 128/30
#
#k = 2 * np.pi / lambDa
#
#kz = np.sqrt(k ** 2 - kx ** 2 - ky ** 2)
#
d = 2

coef = Et_close / Et_distance


Et_df = np.fft.fftshift(np.fft.fft2(Et_distance))
Et_cf = np.fft.fftshift(np.fft.fft2(Et_close))

coef_f = Et_cf / Et_df
for i in range(np.shape(coef_f)[0]):
    for j in range(np.shape(coef_f)[0]):
        if np.abs(coef_f[i, j]) >= 4 or np.abs(coef_f[i, j]) <= -4:
            coef_f[i, j] = 0

phase_shift = np.exp(1j * d * kz)
for i in range(np.shape(phase_shift)[0]):
    for j in range(np.shape(phase_shift)[0]):
        if kz[i, j] != 0:
            phase_shift[i, j] /= kz[i, j]

fft_error = coef_f - phase_shift

Et_pf = Et_df * phase_shift
Et_p = np.fft.ifft2(np.fft.ifftshift(Et_pf))
#
print(np.allclose(Et_p, Et_close))

#v_min = -1
#v_max = 0.3

#plt.figure()
#plt.subplot(121)
#plt.imshow(np.real(coef_f))
#plt.colorbar()
#plt.title('Real ratio fft')
#plt.subplot(122)
#plt.imshow(np.imag(coef_f))
#plt.colorbar()
#plt.title('Imag ratio fft')

plt.figure()
plt.subplot(121)
plt.imshow(np.real(phase_shift))
plt.xlabel('kx')
plt.ylabel('ky')
ax = plt.gca();
ax.set_xticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_yticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_xticklabels(kfreq[::48]);
ax.set_yticklabels(kfreq[::48]);
plt.colorbar()
plt.title('Real phase shift fft')
plt.subplot(122)
plt.imshow(np.imag(phase_shift))
plt.colorbar() 
plt.title('Imag phase shift fft')
#
plt.figure()
plt.subplot(331)
plt.imshow(np.real(Et_p))
plt.colorbar()
plt.title('Real propagated')
plt.subplot(332)
plt.imshow(np.imag(Et_p))
plt.colorbar()
plt.title('Imag propagated')
plt.subplot(333)
plt.imshow(np.abs(Et_p))
plt.colorbar()
plt.title('Magnitude propagated')

plt.subplot(334)
plt.imshow(np.real(Et_close))
plt.colorbar()
plt.title('Real E close')
plt.subplot(335)
plt.imshow(np.imag(Et_close))
plt.colorbar()
plt.title('Imag E close')
plt.subplot(336)
plt.imshow(np.abs(Et_close))
plt.colorbar()
plt.title('Magnitude E close')

#
plt.subplot(337)
plt.imshow(np.real(Et_close)-np.real(Et_distance))
plt.colorbar()
plt.title('Real Error')
plt.subplot(338)
plt.imshow(np.imag(Et_close)-np.imag(Et_distance))
plt.colorbar()
plt.title('Imag Error')
plt.subplot(339)
plt.imshow(np.abs(Et_close)-np.abs(Et_distance))
plt.colorbar()
plt.title('Magnitude Error')

#
plt.figure()
plt.subplot(121)
plt.imshow(np.real(fft_error))
plt.title('Real fft coef error')
plt.xlabel('kx')
plt.ylabel('ky')
ax = plt.gca();
ax.set_xticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_yticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_xticklabels(kfreq[::48]);
ax.set_yticklabels(kfreq[::48]);
plt.colorbar()
plt.subplot(122)
plt.imshow(np.imag(fft_error))
plt.title('Imag fft coef error')
plt.xlabel('kx')
plt.ylabel('ky')
ax = plt.gca();
ax.set_xticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_yticks(np.arange(0, np.shape(kfreq)[0], 48));
ax.set_xticklabels(kfreq[::48]);
ax.set_yticklabels(kfreq[::48]);
plt.colorbar()



#plt.figure()
#plt.subplot(121)
#plt.imshow(np.real(kz))
#plt.subplot(122)
#plt.imshow(np.imag(kz))