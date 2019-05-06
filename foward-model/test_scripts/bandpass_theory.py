# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 17:08:02 2019

this script verifies the construction of a bandpass filter

Editor:
    Shihao Ran
    STIM Laboratory
"""
#%%
import numpy as np
import matplotlib.pyplot as plt

#%%
def BPF(halfgrid, simRes, NA_in, NA_out):
    #create a bandpass filter
    #change coordinates into frequency domain
        
    df = 1/(halfgrid*2)
    
    iv, iu = np.meshgrid(np.arange(0, simRes, 1), np.arange(0, simRes, 1))
    
    u = np.zeros(iu.shape)
    v = np.zeros(iv.shape)
    
    #initialize the filter as All Pass
    BPF = np.ones(iv.shape)
    
    idex1, idex2 = np.where(iu <= simRes/2)
    u[idex1, idex2] = iu[idex1, idex2]
    
    idex1, idex2 = np.where(iu > simRes/2)
    u[idex1, idex2] = iu[idex1, idex2] - simRes +1
    
    u *= df
    
    idex1, idex2 = np.where(iv <= simRes/2)
    v[idex1, idex2] = iv[idex1, idex2]
    
    idex1, idex2 = np.where(iv > simRes/2)
    v[idex1, idex2] = iv[idex1, idex2] - simRes +1
    
    v *= df
    
    magf = np.sqrt(u ** 2 + v ** 2)
    
    #block lower frequency
    idex1, idex2 = np.where(magf < NA_in / lambDa)
    BPF[idex1, idex2] = 0
    #block higher frequency
    idex1, idex2 = np.where(magf > NA_out / lambDa)
    BPF[idex1, idex2] = 0
    
    return BPF

#%%
# specify the detailed parameters to build the bandpass filter
lambDa = 1
k = 2 * np.pi / lambDa
NA_in = 0.3
NA_out = 0.9
res = 128
padding = 0
fov = 16

simRes = res * (2 * padding + 1)
halfgrid = np.ceil(fov / 2) * (padding * 2 + 1)

bpf_truth = BPF(halfgrid, simRes, NA_in, NA_out)

#%%
# basically, a bandpass filter is just a circular mask
# with inner and outer diamater specified by the 
# in and out NA
f_x = np.fft.fftfreq(res, fov/res)

fx, fy = np.meshgrid(f_x, f_x)

fxfy = np.sqrt(fx ** 2 + fy ** 2)

bpf_test = np.zeros((res, res))

mask_out = fxfy <= NA_out
mask_in = fxfy > NA_in

mask = np.logical_and(mask_out, mask_in)

bpf_test[mask] = 1

#%%
plt.figure()
plt.subplot(121)
plt.imshow(np.fft.fftshift(bpf_test))
plt.subplot(122)
plt.imshow(bpf_truth)