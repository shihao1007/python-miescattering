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

#%%
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

#%%
class planewave():
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    def __init__ (self, k, E):
        
        #self.phi = phi
        self.k = k/np.linalg.norm(k)                      
        self.E = E
        
        #force E and k to be orthogonal
        if ( np.linalg.norm(k) > 1e-15 and np.linalg.norm(E) >1e-15):
            s = np.cross(k, E)              #compute an orthogonal side vector
            s = s / np.linalg.norm(s)       #normalize it
            Edir = np.cross(s, k)              #compute new E vector which is orthogonal
            self.k = k
            self.E = Edir / np.linalg.norm(Edir) * np.linalg.norm(E)
    
    def __str__(self):
        return str(self.k) + "\n" + str(self.E)     #for verify field vectors use print command

    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z     #phase term k*r
        ex = np.exp(1j * k_dot_r)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        Ef = self.E.reshape((3, 1, 1)) * ex
        return Ef


def idhf(simFov, simRes, y):
    """
    Inverse Discrete Hankel Transform of an 1D array
    param:
        simFov
        simRes
        y
    return:
        F
        F_x
    """
    
    X = int(simFov/2)
    n_s = int(simRes/2)
    
    # order of the bessel function
    order = 0
    # root of the bessel function
    jv_root = sp.special.jn_zeros(order, n_s)
    jv_M = jv_root[-1]
    jv_m = jv_root[:-1]
#    jv_mX = jv_m/X
    
#    F_term = np.interp(jv_mX, x, y)
    F_term = y[1:]
    # inverse DHT
    F = np.zeros(n_s, dtype=np.complex128)
    jv_k = jv_root[None,...]

    prefix = 2/(X**2)

    Jjj = jv_m[...,None]*jv_k/jv_M
    numerator = sp.special.jv(order, Jjj)
    denominator = sp.special.jv(order+1, jv_m[...,None])**2
    
    summation = np.sum(numerator / denominator * F_term[:-1][...,None], axis=0)
        
    F = prefix * summation
    
    F_x = jv_root*X/jv_M
    
    return F, F_x

def new_bpf(simFov, simRes, NA_in, NA_out):
    # basically, a bandpass filter is just a circular mask
    # with inner and outer diamater specified by the 
    # in and out NA
    f_x = np.fft.fftfreq(simRes, simFov/simRes)
    
    fx, fy = np.meshgrid(f_x, f_x)
    
    fxfy = np.sqrt(fx ** 2 + fy ** 2)
    
    bpf_test = np.zeros((simRes, simRes))
    
    mask_out = fxfy <= NA_out
    mask_in = fxfy >= NA_in
    
    mask = np.logical_and(mask_out, mask_in)
    
    bpf_test[mask] = 1
    
    return bpf_test

#%%
def cal_Et_far_field(a, n, k, k_dir, res, fov, working_dis, scale_factor, NA_in, NA_out):
# calculate the scattering field through far field simulation
    
    # the maximal order
    l_max = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    l = np.arange(0, l_max+1, 1)
    
    # calculate B coefficient
    B = coeff_b(l, k, n, a)
    
    # construct the evaluate plane
    
    simRes = res*(2*padding + 1)
    simFov = fov*(2*padding + 1)
    center = int(simRes/2)
    # halfgrid is the size of a half grid
    halfgrid = np.ceil(simFov/2)
    # range of x, y
    gx = np.linspace(-halfgrid, +halfgrid, simRes)[:center+1]
    gy = gx[0]     
    # make it a plane at z = 0 (plus the working distance) on the Z axis
    z = working_dis
    
    # calculate the distance matrix
    rMag = np.sqrt(gx**2+gy**2+z**2)
    kMag = 2 * np.pi / lambDa
    # calculate k dot r
    kr = kMag * rMag
    
    # calculate the asymptotic form of hankel funtions
    hlkr_asym = np.zeros((kr.shape[0], l.shape[0]), dtype = np.complex128)
    for i in l:
        hlkr_asym[..., i] = np.exp(1j*(kr-i*math.pi/2))/(1j * kr)
    
    # calculate the legendre polynomial
    # get the frequency components
    fx = np.fft.fftfreq(simRes, simFov/simRes)[:center+1]
    fy = fx[0]
    
    # calculate the sum of kx ky components so we can calculate 
    # cos_theta in the Fourier Domain later
    kxky = fx ** 2 + fy ** 2
    # create a mask where the sum of kx^2 + ky^2 is 
    # bigger than 1 (where kz is not defined)
    mask = kxky > 1
    # mask out the sum
    kxky[mask] = 0
    # calculate cos theta in Fourier domain
    cos_theta = np.sqrt(1 - kxky)
    cos_theta[mask] = 0
    # calculate the Legendre Polynomial term
    pl_cos_theta = sp.special.eval_legendre(l, cos_theta[..., None])
    # mask out the light that is propagating outside of the objective
    pl_cos_theta[mask] = 0
    
    # calculate the prefix alpha term
    alpha =(2*l + 1) * 1j ** l
    
    # calculate the matrix besides B vector
    scatter_matrix = hlkr_asym * pl_cos_theta * alpha
    # calculate every order of the integration
    Sum = scatter_matrix * B
    # integrate through all the orders to get the farfield in the Fourier Domain
    E_scatter_fft = np.sum(Sum, axis = -1) * scale_factor
    
    bpf = new_bpf(simFov, simRes, NA_in, NA_out)
    bpf_line = bpf[0, :center+1]
    
    E_scattering, Ex = idhf(simFov, simRes, E_scatter_fft*bpf_line)
    
#    Ei_obj = planewave(k_dir, E)
#    Ei = Ei_obj.evaluate(x, y, np.zeros((simRes, simRes,)))
#    E_incident = Ei[0, ...]
    Et = E_scattering + 1
    
    return Et, E_scattering, Ex

#%%
# set the size and resolution of both planes
fov = 16                    # field of view
res = 128                   # resolution
a = 1                       # radius of the spere
lambDa = 1                  # wavelength
n = 1.5 + 1j*0.01            # refractive index
k = 2 * math.pi / lambDa    # wavenumber
padding = 3                 # padding
working_dis = 10000 * (2 * padding + 1)           # working distance
scale_factor = working_dis * 2 * math.pi * res/fov            # scale factor of the intensity
NA_in = 0.3
NA_out = 0.6

ps = [0, 0, 0]              # position of the sphere
k_dir = [0, 0, -1]          # propagation direction of the plane wave
E = [1, 0, 0]               # electric field vector

#%%
E_t, E_near_line, Ex = cal_Et_far_field(a, n, k, k_dir,
                                         res, fov,
                                         working_dis, scale_factor,
                                         NA_in, NA_out)
simRes = res * (2 * padding + 1)
simFov = fov * (2 * padding + 1)
halfgrid = np.ceil(simFov/2)
#bpf = BPF(halfgrid, simRes, NA_in, NA_out)
#
#E_t_bandpass = imgAtDetec(E_t, bpf)
#
#fx_axis = np.fft.fftshift(np.fft.fftfreq(simRes, simFov/simRes))
#
##%%
#plt.figure()
#plt.set_cmap('RdYlBu')
#plt.subplot(231)
#plt.imshow(np.real(E_scatter_fftshift), extent = [fx_axis[0], fx_axis[-1], fx_axis[0], fx_axis[-1]])
#plt.title('Fourier Domain, Real')
#plt.colorbar()
#
#plt.subplot(232)
#plt.imshow(np.imag(E_scatter_fftshift), extent = [fx_axis[0], fx_axis[-1], fx_axis[0], fx_axis[-1]])
#plt.title('Fourier Domain, Imaginary')
#plt.colorbar()
#
#plt.subplot(233)
#plt.imshow(np.abs(E_scatter_fftshift), extent = [fx_axis[0], fx_axis[-1], fx_axis[0], fx_axis[-1]])
#plt.title('Fourier Domain, Abs')
#plt.colorbar()
#
#plt.subplot(234)
#plt.imshow(np.real(E_scattering), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
#plt.title('Scatter Field, Real')
#plt.colorbar()
#
#plt.subplot(235)
#plt.imshow(np.imag(E_scattering), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
#plt.title('Scatter Field, Imaginary')
#plt.colorbar()
#
#plt.subplot(236)
#plt.imshow(np.abs(E_scattering), extent=[-simFov/2, simFov/2, -simFov/2, simFov/2])
#plt.title('Scatter Field, Abs')
#plt.colorbar()