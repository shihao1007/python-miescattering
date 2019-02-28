# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:14:36 2019

back propagation verification code

Editor:
    Shihao Ran
    STIM Laboratory
"""
import numpy as np
import matplotlib.pyplot as plt

#compute the coordinates grid in Fourier domain for the calculation of
#corresponding phase shift value at each pixel
#return the frequency components at z axis in Fourier domain
def cal_kz():
    #the coordinates in Fourier domain is constructed from the coordinates in
    #spatial domain, specifically,
    #1. Get the pixel size in spatial domain, P_size = FOV / Image_Size
    #2. Fourier domain size, F_size = 1 / P_size
    #3. Make a grid with [-F_size / 2, F_size / 2, same resolution]
    #4. Pixel size in Fourier domain will be 1 / Image_size
    
    kx = 2 * np.pi * 128 / k[0]
    ky = 2 * np.pi * 128 / k[1]
    
    kz = np.sqrt(1 - kx ** 2 - ky ** 2)
    
    #return it
    return kz


#propogate the field with the specified frequency components and distance
    # Et: the field in spatial domain to be propagated
    # k_z: frequency component in z axis
    # l: distance to propagate
def propagate(Et, k_z, l):
    
    #compute the phase mask for shifting each pixel of the field
    phaseMask = np.exp(1j * k_z * l)
    
    #Fourier transform of the field and do fft-shift to the Fourier image
    #so that the center of the Fourier transform is at the origin
    E_orig = Et
    fE_orig = np.fft.fft2(E_orig)
    fE_shift = np.fft.fftshift(fE_orig)
    
    #apply phase shift to the field in Fourier domain
    fE_propagated = fE_shift * phaseMask
    
    #inverse shift the image in Fourier domain
    #then apply inverse Fourier transform the get the spatial image
    fE_inverse_shift = np.fft.ifftshift(fE_propagated)
    E_prop = np.fft.ifft2(fE_inverse_shift)
    
    #return the propagated field
    return E_prop

lambDa = 2 * np.pi
d = 5
kz = cal_kz()

Et_prop = propagate(Et_distance, kz, -d)

#E_d *= Emask1

plt.figure()
plt.subplot(3, 2, 1)
plt.imshow(np.real(Et_prop))
plt.title('Real of prop E')
plt.axis('off')
plt.colorbar()

plt.subplot(3, 2, 2)
plt.imshow(np.imag(Et_prop))
plt.title('Imaginary of prop E')
plt.axis('off')
plt.colorbar()

plt.subplot(3, 2, 3)
plt.imshow(np.real(Et_close)) 
plt.title('Real of E_0')
plt.axis('off')
plt.colorbar()

plt.subplot(3, 2, 4)
plt.imshow(np.imag(Et_close))
plt.title('Imaginary of E_0')
plt.axis('off')
plt.colorbar()

plt.subplot(3, 2, 5)
plt.imshow(np.real(Et_prop)-np.real(Et_close))
plt.title('Real of error')
plt.axis('off')
plt.colorbar()

plt.subplot(3, 2, 6)
plt.imshow(np.imag(Et_prop)-np.imag(Et_close))
plt.title('Imaginary of error')
plt.axis('off')
plt.colorbar()
