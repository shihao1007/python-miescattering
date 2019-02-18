# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:17:10 2018

@author: shihao
"""

import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
import scipy as sp
import math
from scipy.io import loadmat
from matplotlib import animation as animation
import time

class mieScattering:
    
    def __init__(self, n, res, a, numSample, NA):
        self.n = n
        self.res = res
        self.a = a
        self.numSample = numSample
        #parameters used to calculate the fields
        #resolution
        #res = 150
        #position of the sphere
        self.ps = np.asarray([0, 0, 0])
        #position of the focal point
        self.pf = np.asarray([0, 0, 0])
        
        #padding for displaying the figure
        self.padding = 1
        #amplitude of the incoming electric field
        self.E0 = 1
        #in and out numerical aperture of the condenser
        self.NA_in = 0
        self.NA_out = NA
        #theta and phi in spherical coordinate system
        self.theta = 1.5708
        self.phi = 0
        #pixel size of the figure
        self.pixel = 1.1
        
        self.alpha1 = math.asin(self.NA_in)
        self.alpha2 = math.asin(self.NA_out)
        
        #convert coordinates to cartesian if necessary
#        x, y, z = self.sph2cart(self.theta, self.phi, 1)
        
        #specify the direction of the incoming light
        self.lightdirection = np.asarray([0, 0, -1])
        
        #specify the wavelength of the incident light
        self.lambDa = 2.8554
        #refractive index of the sphere
        #n = 1.4763 + 0.000258604 * 1j
        
        #radius of the sphere
        #a = 15
        
        
        #magnitude of the k vector
        #wavenumber = magk = 2*np.pi/lambDa
        self.magk = 2.2004
        self.kVec = self.lightdirection * self.magk
        
        #field of view, the size of the pic
        self.fov = np.ceil(self.res*self.pixel)
        #simulation resolution
        self.simRes = self.res*(2*self.padding + 1)
        
        #initialize a plane to evaluate the field
        self.halfgrid = np.ceil(self.fov/2)*(2*self.padding +1)
        gx = np.linspace(-self.halfgrid, +self.halfgrid-1, self.simRes)
        gy = gx
        
        #if it is a plane in []
        [self.y, self.z] = np.meshgrid(gx, gy)
        
        #make it a plane at 0 on the axis below
        self.x = np.zeros((self.simRes, self.simRes,))
        
        #initialize r vectors in the space
        self.rVecs = np.zeros((self.simRes, self.simRes, 3))
        
        self.rVecs[:,:,0] = self.x
        self.rVecs[:,:,1] = self.y
        self.rVecs[:,:,2] = self.z
        
        #compute the rvector relative to the sphere
        self.rVecs_ps = self.rVecs - self.ps
        
        self.bpf = self.BPF(self.halfgrid, self.simRes, self.NA_in, self.NA_out)

    def sampled_kvectors_spherical_coordinates(self, NA, NumSample, kd):
    #sample multiple planewaves at different angle to do simulation as a focused beam
        # return a list of planewave direction vectors Kd
        # and the corresponding scale factor
        # NA: numberical aperture of the lens which planewaves are focused from
        # NumSample: number of samples(planewaves) along longitudinal and latitudinal axis
        # kd: center planewave of the focused beam
        
        #allocatgge space for the field and initialize it to zero
        CenterKd = [0, 0, -1]                                       #defualt planewave coming in perpendicular to the surface
        kd = kd / np.linalg.norm(kd)                                #normalize the new planewave
        r = np.sqrt(CenterKd[0] ** 2 + CenterKd[1] ** 2 + CenterKd[2] ** 2)             #radiance of the hemisphere where the k vectors are sampled from
        
        if(kd[0] == CenterKd[0] and kd[1] == CenterKd[1] and kd[2] == CenterKd[2]):     #if new planewave is at the same direction as the default plane wave
            rotateAxis = CenterKd                                   #set rotation axis as defualt k vector
            RoAngle = 0                                             #set rotation axis as 0 degrees
        else:                                                       #if new plane wave is at different direction as the defualt planewave, rotation is needed
            rotateAxis = np.cross(CenterKd, kd)                     #find a axis which is perpendicular to both vectors to be rotation axis
            RoAngle = math.asin(kd[2] / r)                          #calculate the rotation angle
        beamRotate = Quaternion(axis=rotateAxis, angle=RoAngle)     #create a quaternion for rotation
        
        Kdt = np.zeros((NumSample, 3, NumSample))                   #initialize the planewave list
         
        scaleFactort = np.zeros((NumSample, NumSample))             #initialize a list of scalefactors which are used to scale down the amplitude of planewaves later on along latitude domain
        
        
        #convert the axis from Cartesian to Spherical
        if(CenterKd[0] == 0 or CenterKd[1] == 0):                   #if the defualt planewave is at the direction of Z axis
            theta = 0                                               #set azimuthal angle theta as 0
        else:
            theta = math.atan(CenterKd[1] / CenterKd[0])            #if not calculate theta from X and Y coordinates
        
        pha = math.acos(CenterKd[2] / r)                            #calculate polar angle pha from Z coordinate
        
        phaM = math.asin(NA / np.real(n))                                   #calculate sample range of pha from numerical aperture
        thetaM = 2* np.pi                                           #set sample range of theta as 2pi
        phaStep = phaM / NumSample                                  #set longitudinal sample resolution as maximal pha divided by number of samples
        thetaStep = thetaM / NumSample                              #set latitudinal sample resolution as maximal theta divided by number of samples
        for i in range(NumSample):                                  #sample along longitudinal (pha) domain
            for j in range(NumSample):                              #sample along latitudinal (theta) domain
                KdR = r                                             #sample hemisphere radiance will be all the same as r
                KdTheta = theta + thetaStep * j                     #sample theta at each step in the sample range
                KdPha = pha + phaStep * i                           #sample theta at each step in the sample range
                
                Kdt[j,0,i] = KdR * np.cos(KdTheta) * np.sin(KdPha)   #convert coordinates from spherical to Cartesian
                Kdt[j,1,i] = KdR * np.sin(KdTheta) * np.sin(KdPha)
                Kdt[j,2,i] = KdR * np.cos(KdPha)
                Kdt[j,:,i] = beamRotate.rotate(Kdt[j,:,i])            #rotate k vectors by the quaternion generated
                
                scaleFactort[j,i] = np.sin(KdPha)                    #calculate the scalefactors by the current polar angle pha
        
        return Kdt, scaleFactort
    
    
    def Legendre(self, order, x):
    #calcula order l legendre polynomial
            #order: total order of the polynomial
            #x: array or vector or scalar for the polynomial
            #return an array or vector with all the orders calculated
            
        if np.isscalar(x):
        #if x is just a scalar value
        
            P = np.zeros((order+1, 1))
            P[0] = 1
            if order == 0:
                return P
            P[1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[j+1] = ((2*j+1)/(j+1)) *x *(P[j]) - ((j)/(j+1))*(P[j-1])
            return P
        
        elif np.asarray(x).ndim == 1:
        #if x is a vector
            P = np.zeros((len(x), order))
            P[:,0] = 1
            if order == 0:
                return P
            P[:, 1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[:,j+1] = ((2*j+1)/(j+1)) *x *(P[:, j]) - ((j)/(j+1))*(P[:, j-1])
            return P
        
        else:
        #if x is an array
            P = np.zeros((x.shape + (order+1,)))
            P[..., 0] = 1
            if order == 0:
                return P
            P[..., 1] = x
            if order == 1:
                return P
            for j in range(1, order):
                P[..., j+1] = ((2*j+1)/(j+1)) *x *(P[..., j]) - ((j)/(j+1))*(P[..., j-1])
            return P
        
        
    def sph2cart(self, az, el, r):
    #convert coordinates from spherical to cartesian
            #az: azimuthal angle, horizontal angle with x axis
            #el: polar angle, vertical angle with z axis
            #r: radial distance with origin
            
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z
    
    
    def sphbesselj(self, order, x, mode):
    #calculate the spherical bessel function of the 1st kind with order specified
        #order: the order to be calculated
        #x: the variable to be calculated
        #mode: 1 stands for prime, -1 stands for derivative, 0 stands for nothing
            if np.isscalar(x):
                return np.sqrt(np.pi / (2*x)) * sp.special.jv(order + 0.5 + mode, x)
            
            elif np.asarray(x).ndim == 1:
                ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
                for i in range(len(order)):
                    ans[:,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
                return ans
            
            else:
                ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
                for i in range(len(order)):
                    ans[...,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
                return ans
            
            
            
    def sphhankel(self, order, x, mode):
    #general form of calculating spherical hankel functions of the first kind at x
        
        if np.isscalar(x):
            return np.sqrt(np.pi / (2*x)) * (sp.special.jv(order + 0.5 + mode, x) + 1j * sp.special.yv(order + 0.5 + mode, x))
    #
            
        elif np.asarray(x).ndim == 1:
            ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
            for i in range(len(order)):
                ans[:,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
            return ans
        else:
            ans = np.zeros((x.shape + (len(order),)), dtype = np.complex128)
            for i in range(len(order)):
                ans[...,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
            return ans
        
    
    #derivative of the spherical bessel function of the first kind
    def derivSphBes(self, order, x):
        js_n = np.zeros(order.shape, dtype = np.complex128)
        js_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
        js_n_p_1 = np.zeros(order.shape, dtype = np.complex128)
        
        js_n = self.sphbesselj(order, x, 0)
        js_n_m_1 = self.sphbesselj(order, x, -1)
        js_n_p_1 = self.sphbesselj(order, x, 1)
        
        j_p = 1/2 * (js_n_m_1 - (js_n + x * js_n_p_1) / x)
        return j_p
    
    #derivative of the spherical hankel function of the first kind
    def derivSphHan(self, order, x):
        sh_n = np.zeros(order.shape, dtype = np.complex128)
        sh_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
        sh_n_p_1 = np.zeros(order.shape, dtype = np.complex128)
    
        sh_n = self.sphhankel(order, x, 0)
        sh_n_m_1 = self.sphhankel(order, x, -1)
        sh_n_p_1 = self.sphhankel(order, x, 1)
        
        h_p = 1/2 * (sh_n_m_1 - (sh_n + x * sh_n_p_1) / x)
        return h_p
    
        
    def calFocusedField(self, simRes, magk, rMag):
    #calculate a focused beam from the paramesters specified
        #the order of functions for calculating focused field
        orderEf = 100
        #il term
        ordVec = np.arange(0, orderEf+1, 1)
        il = 1j ** ordVec
        
        #legendre polynomial of the condenser
        plCosAlpha1 = self.Legendre(orderEf+1, np.cos(self.alpha1))
        plCosAlpha2 = self.Legendre(orderEf+1, np.cos(self.alpha2))
        
        #initialize magnitude of r vector at each pixel
        rMag = np.zeros((simRes, simRes))
        #initialize angle between k vector to each r vector 
        cosTheta = np.zeros((rMag.shape))
        #initialize normalized r vector
        rNorm = np.zeros((self.rVecs.shape))
        #normalized k vector 
        kNorm = self.kVec / magk
        #compute rMag and rNorm and cosTheta at each pixel
        
        rMag = np.sqrt(np.sum(self.rVecs_ps**2, 2))
        rNorm = self.rVecs_ps / rMag[...,None]
        cosTheta = np.dot(rNorm, kNorm)
        
        
        #compute spherical bessel function at kr
        jlkr= self.sphbesselj(ordVec, magk*rMag, 0)
        
        #compute legendre polynomial of all r vector
        plCosTheta = self.Legendre(orderEf, cosTheta)
        
        #product of them
        jlkrPlCosTheta = jlkr * plCosTheta
        
        il = il.reshape((1, 1, orderEf+1))
        iljlkrplcos = jlkrPlCosTheta * il
        
        order = 0
        iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[0]+plCosAlpha2[0])
        
        order = 1
        iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[0]+plCosAlpha2[0])
        
        for order in range(2, orderEf):
            iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[order-1]+plCosAlpha2[order-1])
        
        #sum up all orders
        Ef = 2*np.pi*self.E0*np.sum(iljlkrplcos, axis = 2)
        
        return Ef
    
    def scatterednInnerField(self, lambDa, magk, n):
        start2 = time.time()
        #calculate and return a focused field and the corresponding scattering field and internal field
        #maximal number of orders used to calculate Es and Ei
        numOrd = math.ceil(2*np.pi * self.a / lambDa + 4 * (2 * np.pi * self.a / lambDa) ** (1/3) + 2)
        #create an order vector
        ordVec = np.arange(0, numOrd+1, 1)
        #calculate the prefix term (2l + 1) * i ** l
        twolplus1 = 2 * ordVec + 1
        il = 1j ** ordVec
        twolplus1_il = twolplus1 * il
        #compute the arguments for spherical bessel functions, hankel functions and thier derivatives
        ka = magk * self.a
        kna = magk * n * self.a
        #number of samples
        
        
        #evaluate the spherical bessel functions of the first kind at ka
        jl_ka = self.sphbesselj(ordVec, ka, 0)
        
        #evaluate the derivative of the spherical bessel functions of the first kind at kna
        jl_kna_p = self.derivSphBes(ordVec, kna)
        
        #evaluate the spherical bessel functions of the first kind at kna
        
        jl_kna = self.sphbesselj(ordVec, kna, 0)
        
        #evaluate the derivative of the spherical bessel functions of the first kind of ka
        jl_ka_p = self.derivSphBes(ordVec, ka)
        
        #compute the numerator for B coefficients
        numB = jl_ka * jl_kna_p * n - jl_kna * jl_ka_p
        
        #evaluate the hankel functions of the first kind at ka
        hl_ka = self.sphhankel(ordVec, ka, 0)
        
        #evaluate the derivative of the hankel functions of the first kind at ka
        hl_ka_p = self.derivSphHan(ordVec, ka)
        
        #compute the denominator for coefficient A and B
        denAB = jl_kna * hl_ka_p - hl_ka * jl_kna_p * n
        
        #compute B
        B = np.asarray(twolplus1_il * (numB / denAB), dtype = np.complex128)
        B = np.reshape(B, (1, 1, numOrd + 1))
        
        #compute the numerator of the scattering coefficient A
        numA = jl_ka * hl_ka_p - jl_ka_p * hl_ka
        
        #compute A
        A = np.asarray(twolplus1_il * (numA / denAB), dtype = np.complex128)
        A = np.reshape(A, (1, 1, numOrd + 1))
        
        #compute the distance between r vector and the sphere      
        rMag = np.sqrt(np.sum(self.rVecs_ps ** 2, 2))
        rNorm = self.rVecs_ps / rMag[..., None]
        #computer k*r term
        kr = magk * rMag
        
        #compute the spherical hankel function of the first kind for kr
        hl_kr = self.sphhankel(ordVec, kr, 0)
        
        #computer k*n*r term
        knr = kr * n
        
        #compute the spherical bessel function of the first kind for knr
        jl_knr = self.sphbesselj(ordVec, knr, 0)
        
        #compute the distance from the center of the sphere to the focal point/ origin
        #used for calculating phase shift later
        c = self.ps - self.pf
        
        #initialize Ei and Es field
        Ei = np.zeros((self.simRes, self.simRes), dtype = np.complex128)
        Es = np.zeros((self.simRes, self.simRes), dtype = np.complex128)
        
        #a list of sampled k vectors
        k_jt, scalet = self.sampled_kvectors_spherical_coordinates(self.NA_out, self.numSample, self.lightdirection)
        
        cos_thetat = np.dot(rNorm, k_jt)
        pl_costhetat = self.Legendre(numOrd, cos_thetat)
        
        hlkr_plcosthetat = hl_kr[:,:,None,None,:] * pl_costhetat
        
        jlknr_plcosthetat = jl_knr[:,:,None,None,:] * pl_costhetat
        phaset = np.exp(1j * magk * np.dot(c, k_jt))
        
        Es = np.sum(np.sum((phaset * scalet)[None, None, ...] * np.sum(hlkr_plcosthetat * B, axis = -1), axis = -1), axis = -1)
        Ei = np.sum(np.sum((phaset * scalet)[None, None, ...] * np.sum(jlknr_plcosthetat * A, axis = -1), axis = -1), axis = -1)
        
        Es[rMag<self.a] = 0
        Ei[rMag>=self.a] = 0
        
        Ef = self.calFocusedField(self.simRes, self.magk, rMag)
        
        
        Etot = np.zeros((self.simRes, self.simRes), dtype = np.complex128)
        Etot[rMag<self.a] = Ei[rMag<self.a]
        Etot[rMag>=self.a] = Es[rMag>=self.a] + Ef[rMag>=self.a]
    
    #    plt.figure()
    #    #plt.plot(np.abs(hl_ka))
    #    plt.imshow(np.log10(np.abs(Etot)))
        end2 = time.time()
        print("render field: " + str(start2 - end2) + " s\n")
        return Ef, Etot
    
    def BPF(self, halfgrid, simRes, NA_in, NA_out):
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
        idex1, idex2 = np.where(magf < NA_in / self.lambDa)
        BPF[idex1, idex2] = 0
        #block higher frequency
        idex1, idex2 = np.where(magf > NA_out / self.lambDa)
        BPF[idex1, idex2] = 0
        
        return BPF
    
    def imgAtDetec(self, Etot, Ef):
        #2D fft to the total field
        Et_d = np.fft.fft2(Etot)
        Ef_d = np.fft.fft2(Ef)
        
        #apply bandpass filter to the fourier domain
        Et_d *= self.bpf
        Ef_d *= self.bpf
        
        #invert FFT back to spatial domain
        Et_bpf = np.fft.ifft2(Et_d)
        Ef_bpf = np.fft.ifft2(Ef_d)
        
        #initialize cropping
        cropsize = self.padding * self.res
        startIdx = int(np.fix(self.simRes /2 + 1) - np.floor(cropsize/2))
        endIdx = int(startIdx + cropsize - 1)
        
        #save the field
    #    np.save(r'D:\irimages\irholography\New_QCL\BimSimPython\Et15YoZ.npy', Et_bpf)
        
        #crop the image
        D_Et = np.zeros((cropsize, cropsize), dtype = np.complex128)
        D_Et = Et_bpf[startIdx:endIdx, startIdx:endIdx]
        D_Ef = np.zeros((cropsize, cropsize), dtype = np.complex128)
        D_Ef = Ef_bpf[startIdx:endIdx, startIdx:endIdx]
    
        return [D_Et, D_Ef]

def varyingN():
    n = 1.5 #+ 0.000258604 * 1j
    N = [n - i*0.1 for i in range(6)]
    D_Et_all = [mieScattering(N[i], 150, 30)[0] for i in range(6)]
    plt.figure()
    for i in range(6):
        plt.subplot(6, 3, i*3 + 1)
        plt.title("real")
        plt.imshow(np.real(D_Et_all[i]))
    
        
        plt.subplot(6, 3, i*3 + 2)
        plt.title("imaginary")
        plt.imshow(np.imag(D_Et_all[i]))
        
        plt.subplot(6, 3, i*3 + 3)
        plt.title("magnitude")
        plt.imshow(np.abs(D_Et_all[i]))


n = 1.5 #+ 0.000258604 * 1j
res = 20
a = 2
numSample = 10
NA = 0.3

start1 = time.time()
MSI = mieScattering(n, res, a, numSample, NA)
end1 = time.time()
print("create object: " + str(start1 - end1) + " s\n")


Ef, Etot = MSI.scatterednInnerField(MSI.lambDa, MSI.magk, MSI.n)
D_Et, D_Ef = MSI.imgAtDetec(Etot, Ef)




plt.figure()
plt.subplot(131)
plt.title("real")
plt.imshow(np.real(D_Et))


plt.subplot(132)
plt.title("imaginary")
plt.imshow(np.imag(D_Et))

plt.subplot(133)
plt.title("magnitude")
plt.imshow(np.abs(D_Et))

#np.save(r'D:\irimages\irholography\New_QCL\BimSimPython\Et30XOY.npy', D_Et)

#plt.title("YOZ Plane")
#plt.axis("off")
#plt.set_cmap('gnuplot2')
#plt.colorbar()
#


#temp = loadmat(r'D:\irimages\irholography\oldQCL\bimsim_test\EsEi\DEt.mat')
##temp = loadmat(r'D:\irimages\irholography\bimsim_test\DEt.mat')
#DEt= temp["D_Et"]
##print(np.amax(sh11 - hl_kr[:,:,0]))
##
##diff = np.zeros(hl_kr.shape, dtype = np.complex128)
##for i in range(numOrd):
##    diff[:,:,i] = cpu_hl_kr[:,:,i] - hl_kr[:,:,i]
##    print(np.amax(cpu_hl_kr[:,:,i] - hl_kr[:,:,i]))
##
###
####
#plt.figure()
###plt.plot(np.abs(hl_ka))
#plt.imshow(DEt)
#plt.plot(diff)
##plt.colorbar()
##plt.imshow(np.abs(Etot))
##
#_min, _max = np.amin(np.imag(hl_kr)), np.amax(np.imag(hl_kr))
#fig = plt.figure()
#
#img = []
#for i in range(23):
#    img.append([plt.imshow(np.imag(diff[:,:,i]), vmax = _max, vmin = _min)])
#
#ani = animation.ArtistAnimation(fig,img,interval=100)
#writer = animation.writers['ffmpeg'](fps=10)
#plt.colorbar()


#alpha2m= t["alpha2"]
####
####
#plt.figure()
#plt.subplot(121)
#plt.imshow(np.abs(iljlkr1[:,:,1]))
#plt.colorbar()
#plt.title("BimSIm")
#plt.subplot(122)
#plt.imshow(np.abs(iljlkrplcos[:,:,1]))
#plt.colorbar()
#plt.title("Shihao")
#
#plt.figure()
#plt.plot(alpha1m)
#plt.plot(plCosAlpha1)


