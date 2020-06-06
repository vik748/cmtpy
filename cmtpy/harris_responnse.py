#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 21:27:20 2020

@author: vik748
"""
from scipy.ndimage import convolve
import numpy as np
import cv2

def fspecial_gauss(size, sigma):
    """
    Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

def xy_gradients(img):
    '''
    Return x and y gradients of an image. Similar to np.gradient
    '''
    kernelx = 1/2*np.array([[-1,0,1]])
    kernely = 1/2*np.array([[-1],[0],[1]])
    fx = cv2.filter2D(img,cv2.CV_32F,kernelx)
    fy = cv2.filter2D(img,cv2.CV_32F,kernely)
    return fy, fx

#def eigen_image_p(self,lpf,scale):
    '''
    ef2,nL = eigen_image_p(lpf,scale)
    set up in pyramid scheme with detection scaled smoothed images
    ef2 is the interest point eigen image
    lpf smoothed by the detection scale gaussian
    Gi = fspecial('gaussian',ceil(7*sigi),sigi);
    '''

def plot_harris_eig_vals(gr, ax):
    sigd = 1.0          # derivation scale
    sigi = 2.75         # integration scale 1.4.^[0:7];%1.2.^[0:10]

    pyrlpf = fspecial_gauss(int(np.ceil(7*sigd)),sigd)
    Gi = fspecial_gauss(11,sigi)
    lpimg = convolve(gr,pyrlpf,mode='constant')


    [fy,fx] = xy_gradients(lpimg)

    [fxy,fxx] = xy_gradients(fx)
    [fyy,fyx] = xy_gradients(fy)
    nL = np.abs(fxx+fyy)

    Mfxx = convolve(np.square(fx),Gi,mode='constant')
    Mfxy = convolve(fx*fy,Gi,mode='constant')
    Mfyy = convolve(np.square(fy),Gi,mode='constant')

    M = np.zeros((gr.shape[0], gr.shape[1], 2,2))
    M[:,:,0,0] = Mfxx
    M[:,:,0,1] = Mfxy
    M[:,:,1,0] = Mfxy
    M[:,:,1,1] = Mfyy

    eig_vals = np.linalg.eigvals(M)
    ax.plot(eig_vals[:,:,0].ravel(), eig_vals[:,:,1].ravel(),'.',ms = 2)
    return

'''
''Tr = Mfxx+Mfyy
Det = Mfxx*Mfyy-np.square(Mfxy)
with np.errstate(invalid='ignore'):
    sqrterm = np.sqrt(np.square(Tr)-4*Det)

ef2 = 0.5*(Tr - sqrterm)
#return np.nan_to_num(ef2),nL
'''