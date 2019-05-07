# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:34:10 2019

Traditional Mie scattering model where the incident field is a plane wave

Two individual visualization planes are plotted in this demo
By specify the orientation of the planes when initialze 
the miescattering class

Editor:
    Shihao Ran
    STIM Laboratory
"""

#%%
# import packages
import numpy as np
from matplotlib import pyplot as plt
import chis.MieScattering as ms
import chis.traditional_mie as traditional_mie
#%%
# specify parameters
# please refer to the comments in traditional_mie.getTotalField()
# for the meanings of these parameters
# by keep the cursor in the function and hit Ctrl+I
k = [0, 0, -1]
res = 256
numSample = 1000
NA_in = 0
NA_out = 0
numFrames = 70
option = 'Horizontal'
n0 = 1.2
fov = 8
padding = 0
a = 1
pp = 0
ps = [0, 0, 0]
simRes, simFov = ms.pad(res, fov, padding)

#%%
# calculate the horizontal simulation
Et_h, B0, Emask0, rVecs= traditional_mie.getTotalField(k, k, n0, res, a, ps, pp, padding, fov, numSample, NA_in, NA_out, option)
# calculate the vertical simulation
Et_v, B1, Emask1, rVecs_v= traditional_mie.getTotalField(k, k, n0, res, a, ps, pp, padding, fov, numSample, NA_in, NA_out, 'Vertical')

# propagate the horizontal simulation
# the field can be propagated along the k direction for d distance
d = 1
Et_0_p = ms.propagate_2D(simRes, simFov, Et_h, d)

# we can add noise to the simulation
noise_amp = 200
noise_mask = np.random.randint(-noise_amp, noise_amp, size = np.shape(Et_h)) / 1000000 + 1
Et_plus_noise = Et_h * noise_mask

#%%
# plot images
plt.figure()

plt.subplot(2, 2, 1)
plt.imshow(np.real(Et_h))
plt.title('Horizontal Real')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 2)
plt.imshow(np.imag(Et_h))
plt.title('Horizontal Imaginary')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 3)
plt.imshow(np.real(Et_v))
plt.title('Vertical Real')
plt.axis('off')
plt.colorbar()

plt.subplot(2, 2, 4)
plt.imshow(np.imag(Et_v))
plt.title('Vertical Imaginary')
plt.axis('off')
plt.colorbar()

