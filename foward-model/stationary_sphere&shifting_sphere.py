# -*- coding: utf-8 -*-
"""
Created on Fri May 17 10:49:31 2019

Editor:
    Shihao Ran
    STIM Laboratory
"""

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

1-D version

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
def shift_window(E, res, side, shift, dimension=2):
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
        dimension: int, 1 or 2
            dimension of the window
            
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
        x_left = - shift
        x_right = int(res/2) - shift
    elif side == 'right':
        x_right = shift
        x_left = -int(res/2) + shift
    else:
        raise ValueError('Invalid Value for side')
    
    # set the indices
    x_left += int(res/2)
    x_right += int(res/2)
    
    # crop the image
    if dimension == 2:
        E_crop = E[y_lower: y_upper, x_left: x_right]
    elif dimension == 1:
        E_crop = E[x_left: x_right]
    else:
        raise ValueError('Invalid Value for dimension')
    
    return E_crop
#%%
# specify parameters
fov = 32                    # field of view
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
shift2 = 99
#%%
# get 1-D far field
E_far_line = ms.far_field(simRes, simFov, working_dis, a, n, 
                          lambDa, k_dir, scale_factor, dimension=1)
# get 1-D near line without bandpass filtering
E_near_line, E_near_x = ms.idhf(simRes, simFov, E_far_line)

# get 1-D near line with incident plane wave
E_near_line += E0

# contatenate the line to make it symmetrical
E_line = np.concatenate((E_near_line[::-1], E_near_line), axis=0)
E_line_crop = E_line[int(np.ceil(len(E_line)/4)):int(np.ceil(len(E_line)*3/4))]

# get shifting sphere
E_shifting = shift_window(E_line, simRes, 'left', shift1, dimension=1)

# merge two spheres
E_merge = E_line_crop + E_shifting

# crop again
E_out = E_merge[int(np.ceil(len(E_merge)/2)):]
#%%
# create a image stram that contains all shifts
shift_lst = np.linspace(shift1, shift2, shift2-shift1+1)

image_lst = np.zeros((len(shift_lst), 480, 640, 4),
                     dtype=np.complex128)

for shift in shift_lst:
    
    shift = int(shift)
    
    # get left sphere
    E_left = shift_window(E_line, simRes, 'left', shift, dimension=1)
    
    # merge two spheres
    E_merge = E_line_crop + E_left
    
    plt.figure()
    plt.plot(np.real(E_merge))
    plt.savefig('temp.png')
    
    img = plt.imread('temp.png')
    
    # add to image list
    image_lst[shift,...] = img

#%%
# create an animation
double_lst = np.concatenate((image_lst, image_lst[::-1]), axis=0)
#%%
animation.anime(double_lst, 15)
#%%
plt.figure()
plt.set_cmap('RdYlBu')

plt.subplot(131)
plt.plot(np.real(E_line_crop))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Complete Field, Real')
#plt.colorbar()

plt.subplot(132)
plt.plot(np.real(E_shifting))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Left Sphere, Real')
#plt.colorbar()

plt.subplot(133)
plt.plot(np.real(E_merge))
plt.plot(np.real(E_out))#, extent = [-simFov/2, simFov/2, -simFov/2, simFov/2])
plt.title('Two Spheres, Real')
#plt.colorbar()