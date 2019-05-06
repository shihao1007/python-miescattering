# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:02:59 2019

Calculate the scattering field of a sphere

The position of the visualization plane is at the plane where right 
across the center of the sphere at the origin

To avoid back progation, the simulation is done by,
First,  create a far field model in the Fourier domain, which can be done by:
    
    1. Replace the Hankel function of the first kind by its asymptotic form
    2. Transform Legendre polynomial into Fourier Domain

Then, apply an inverse Fourier transform to the far field model to get the
near field scattering field

Editor:
    Shihao Ran
    STIM Laboratory
"""

import numpy as np
import scipy as sp
import scipy.special
import math
import matplotlib.pyplot as plt
import chis.MieScattering as ms


#%%
# set parameters
fov = 32                    # field of view
res = 256                   # resolution
a = 1                       # radius of the spere
lambDa = 1                  # wavelength
n = 1.5 + 1j*0.01            # refractive index
k = 2 * math.pi / lambDa    # wavenumber
padding = 1                 # padding
working_dis = 10000 * (2 * padding + 1)           # working distance
scale_factor = working_dis * 2 * math.pi * res/fov            # scale factor of the intensity
NA_in = 0.0
NA_out = 1.0

simRes, simFov = ms.pad(res, fov, padding)

ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector
E0 = 1

#%%
# get 1-D far field
E_far_line = ms.far_field(simRes, simFov, working_dis, a, n, 
                          lambDa, scale_factor, dimention=1)
# get 1-D near line without bandpass filtering
E_near_line, E_near_x = ms.idhf(simRes, simFov, E_far_line)
# get 1-D bandpass filter
bpf_line = ms.bandpass_filter(simRes, simFov, NA_in, NA_out, dimention=1)
# get 1-D near field and its sample index
E_near_line_bp, E_near_x_bp = ms.apply_filter(simRes, simFov, E_far_line, bpf_line)
# far field with incident plane wave
F = E_near_line_bp + E0

#%%

# get ground truth from a 2-D simulation
# get far field
E_far = ms.far_field(simRes, simFov, working_dis, a, n, lambDa, scale_factor)
# get near field
E_near = ms.far2near(E_far) + E0
# get bandpass filter
bpf = ms.bandpass_filter(simRes, simFov, NA_in, NA_out)
# get bandpassed image
E_bp = ms.apply_filter(simRes, simFov, E_near, bpf)
# get the 1-D line
n = E_bp[int(simRes/2), int(simRes/2):]

#%%
# normalize both of them
F_n = (F-np.mean(F))/np.std(F)
n_n = (n-np.mean(n))/np.std(n)

# sample index for ground truth
x = np.linspace(-simFov/2, simFov/2, simRes)[int(simRes/2):]

#%%
plt.figure()
#plt.subplot(121)
plt.plot(E_near_x_bp, np.real(F_n), label='Transformed')
plt.plot(x, np.real(n_n), label='Ground Truth')
plt.ylabel('Intensity')
plt.xlabel('Field of View/um')
plt.legend()
plt.title('Sphere #1, NA_in=0.3, NA_out=0.6')