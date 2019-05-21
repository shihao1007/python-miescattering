# -*- coding: utf-8 -*-
"""
Created on Tue May 14 09:57:10 2019

simulate the situation where the sphere is not positioned at the origin
two approaches:
    1. change the meshgrid when creating horizontal canvas
    2. make a bigger image then shift the cropping window

approch 1 is better for one-time simulation
approch 2 is better for multi-image generation

here, approch 2 is implemented

Editor:
    Shihao Ran
    STIM Laboratory
"""

# import packages
import numpy as np
import matplotlib.pyplot as plt
import chis.MieScattering as ms
from chis import animation

#%%
def shift_window(E, res, side, shift):
    """
    crop the original image with a shifted window so that
    after cropping, the sphere is no long at the center of the image
    
    Parameters
    ----------
        res: int
            resolution of the original image
        side: string, left or right
            relative position of the sphere
        shift: int
            number of pixels to shift the window
            
    Returns
    -------
        E_crop: complex, 2-D array
            the field after cropping
    """
    
    # initialize the window
    y_upper = int(res/4) + int(res/2)
    y_lower = -int(res/4) + int(res/2)
    
    # shift thte window
    if side == 'left':
        x_left = -int(res/16) - shift
        x_right = int(res*7/16) - shift
    elif side == 'right':
        x_right = int(res/16) + shift
        x_left = -int(res*7/16) + shift
    else:
        raise ValueError('Invalid Value for side')
    
    # set the indices
    x_left += int(res/2)
    x_right += int(res/2)
    
    # crop the image
    E_crop = E[y_lower: y_upper, x_left: x_right]
    
    return E_crop
#%%
# specify parameters
fov = 16                    # field of view
res = 256                   # resolution
a = 1                       # radius of the spere
lambDa = 1                  # wavelength
n = 1.5 + 1j*0.01            # refractive index
k = 2 * np.pi / lambDa    # wavenumber
padding = 1                 # padding

simRes, simFov = ms.pad(res, fov, padding)
working_dis = ms.get_working_dis(padding)
scale_factor = ms.get_scale_factor(res, fov, working_dis)            # scale factor of the intensity
NA_in = 0.0
NA_out = 1.0

ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector
E0 = 1

shift1 = 0
shift2 = int(simRes/8)
#%%
# get far field
E_far = ms.far_field(simRes, simFov, working_dis, a, n, lambDa, k_dir, scale_factor)

# get near field
E_near = ms.far2near(E_far) + E0

# get left sphere
E_left = shift_window(E_near, simRes, 'left', shift2)

# get right sphere
E_right = shift_window(E_near, simRes, 'right', shift2)

# merge two spheres
E_merge = E_left + E_right


#%%
# create a image stram that contains all shifts
shift_lst = np.linspace(shift1, shift2, shift2-shift1+1)

image_lst = np.zeros((len(shift_lst), int(simRes/2), int(simRes/2)),
                     dtype=np.complex128)

for shift in shift_lst:
    
    shift = int(shift)
    
    # get left sphere
    E_left = shift_window(E_near, simRes, 'left', shift)
    
    # get right sphere
    E_right = shift_window(E_near, simRes, 'right', shift)
    
    # merge two spheres
    E_merge = E_left + E_right
    
    # add to image list
    image_lst[shift,...] = E_merge

#%%
# create an animation
double_lst = np.concatenate((image_lst, image_lst[::-1]), axis=0)
#%%
animation.anime(double_lst, 15)
#%%
plt.figure()
plt.set_cmap('RdYlBu')

plt.subplot(141)
plt.imshow(np.real(E_near))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Complete Field, Real')
#plt.colorbar()

plt.subplot(142)
plt.imshow(np.real(E_left))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Left Sphere, Real')
#plt.colorbar()

plt.subplot(143)
plt.imshow(np.real(E_right))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Right Sphere, Real')
#plt.colorbar()

plt.subplot(144)
plt.imshow(np.real(E_merge))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Two Spheres, Real')
#plt.colorbar()