#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:39:08 2020

@author: vik748
"""
#from cmtpy.histogram_warping_ace import HistogramWarpingACE
import cv2
import numpy as np
from matplotlib import pyplot as plt

def compute_image_average_contrast(k, gamma=2.2):
    '''
    Given an image at a given scale, compute the average contrast against 4
    neighbouring pixels

    Parameters
    ----------
    k : MxN image as
        DESCRIPTION.
    gamma : TYPE, optional
        DESCRIPTION. The default is 2.2.

    Returns
    -------
    float
        average contrast of the image pixels
    '''
    if k.dtype != np.uint8:
        raise ValueError("Provided image K is not uint8")
    L = 100 * np.sqrt((k / 255) ** gamma )
    # pad image with border replicating edge values
    L_pad = np.pad(L,1,mode='edge')

    # compute differences in all directions
    left_diff = L - L_pad[1:-1,:-2]
    right_diff = L - L_pad[1:-1,2:]
    up_diff = L - L_pad[:-2,1:-1]
    down_diff = L - L_pad[2:,1:-1]

    # create matrix with number of valid values 2 in corners, 3 along edges and 4 in the center
    num_valid_vals = 3 * np.ones_like(L)
    num_valid_vals[[0,0,-1,-1],[0,-1,0,-1]] = 2
    num_valid_vals[1:-1,1:-1] = 4

    pixel_avgs = (np.abs(left_diff) + np.abs(right_diff) + np.abs(up_diff) + np.abs(down_diff)) / num_valid_vals

    return np.mean(pixel_avgs)

def compute_global_contrast_factor(img):
    '''
    Calculate the global_contrast_factor as per
    "Global contrast factor-a new approach to image contrast" (Matkovic, Kresimir et al., 2005)
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.84.2683&rep=rep1&type=pdf

    Parameters
    ----------
    img : MxNx1 or MxNX3 Image array
        DESCRIPTION.

    Returns
    -------
    gcf : float
        Thte calculated global contrast factor
    '''
    if img.ndim != 2:
        gr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gr = img

    superpixel_sizes = [1, 2, 4, 8, 16, 25, 50, 100, 200]

    gcf = 0

    for i,size in enumerate(superpixel_sizes,1):
        wi =(-0.406385 * i / 9 + 0.334573) * i/9 + 0.0877526
        im_scale = cv2.resize(gr, (0,0), fx=1/size, fy=1/size,
                              interpolation=cv2.INTER_LINEAR)
        avg_contrast_scale = compute_image_average_contrast(im_scale)
        gcf += wi * avg_contrast_scale

    return gcf, None


def compute_rms_contrast(gr, mask=None, no_bits=8, debug=False):
    '''
    Convert image to float and compute its Root Mean Square contrast as the
    standard deviation of intensities

    Parameters
    ----------
    gr : MxN unit8 numpy array
        Input grayscale image.
    no_bits : int, optional
        Bit depth if input image. The default is 8.
    debug : bool, optional
        Show computation plots if true. The default is False.

    Returns
    -------
    float
        The computed RMS contrast.
    '''
    if gr.dtype != np.uint8:
        raise ValueError("Provided image gr is not uint8")
    if gr.ndim != 2:
        raise ValueError("Provided image gr is not grayscale")

    levels = 2 ** no_bits - 1
    gr_float = gr / levels
    if debug:
        if not mask is None:
            gr_float_plot = gr_float * mask
            gr_filt_plot = np.mean(gr_float[mask])*mask
        else:
            gr_float_plot = gr_float
            gr_filt_plot = np.mean(gr_float)*np.ones_like(gr_float)

        fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
        ax_imgs = np.empty((13),dtype=object)
        axes[0].set_title('Raw')
        axes[1].set_title('Image Mean')
        axes[2].set_title('Diff')
        [axi.set_axis_off() for axi in axes.ravel()]

        ax_imgs[0] = axes[0].imshow(gr_float_plot, cmap='gray',vmin=0, vmax=1)        
        ax_imgs[1] = axes[1].imshow(gr_filt_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[2] = axes[2].imshow(np.abs(gr_float_plot - gr_filt_plot), cmap='gray')
        fig.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=.9, wspace=0.00, hspace=0.00)
        [fig.colorbar(ax_img, ax=ax, orientation='horizontal',pad=0.05, shrink=0.9) for ax_img, ax in zip(ax_imgs, axes)]
        
    if not mask is None:
        return np.std(gr_float), np.std(gr_float[mask])
    else:
        return np.std(gr_float), None


def compute_box_filt_contrast(gr, mask=None, kernel_size=11, no_bits=8, debug=False):
    '''
    Convert image to float and compute its RMS local contrast using a box filtered image

    Parameters
    ----------
    gr : MxN unit8 numpy array
        Input grayscale image.
    kernel_size : int, optional
        Size of the box kernel. The default is 11.
    no_bits : int, optional
        Bit depth if input image. The default is 8.
    debug : bool, optional
        Show computation plots if true. The default is False.

    Returns
    -------
    contrast : float
        Returns the computed box filtered RMS local contrast.
    '''
    if gr.dtype != np.uint8:
        raise ValueError("Provided image gr is not uint8")
    if gr.ndim != 2:
        raise ValueError("Provided image gr is not grayscale")
    if not mask is None and mask.dtype is not np.dtype('bool'):
        raise ValueError("Provided mask in not bool")

    if kernel_size % 2 == 0:
        raise ValueError ("Kernel size must be an odd integer")

    levels = 2 ** no_bits - 1
    gr_float = gr.astype(np.float32) / levels

    ignore_border_width = kernel_size // 2

    #kernel = np.ones((kernel_size,kernel_size),np.float32)/kernel_size**2
    #gr_filt = cv2.filter2D(gr_float,-1,kernel, borderType=cv2.BORDER_ISOLATED)
    gr_filt = cv2.boxFilter(gr_float, -1, (kernel_size, kernel_size), borderType=cv2.BORDER_ISOLATED)

    gr_float_cropped = gr_float[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    gr_filt_cropped = gr_filt[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    if not mask is None:
            mask_cropped = mask[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
            diff_masked = gr_float_cropped[mask_cropped] - gr_filt_cropped[mask_cropped]
            contrast_masked = np.sum(diff_masked ** 2) / np.sum(mask_cropped)
            
    contrast = np.sum( (gr_float_cropped - gr_filt_cropped)**2 ) / ( gr_float_cropped.shape[0] * gr_float_cropped.shape[1] )

    if debug:
        if not mask is None:
            gr_float_plot = gr_float * mask
            gr_filt_plot = gr_filt * mask
        else:
            gr_float_plot = gr_float
            gr_filt_plot = gr_filt
            
        fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
        ax_imgs = np.empty((13),dtype=object)
        axes[0].set_title('Raw')
        axes[1].set_title('Box Filtered')
        axes[2].set_title('Diff')
        [axi.set_axis_off() for axi in axes.ravel()]

        ax_imgs[0] = axes[0].imshow(gr_float_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[1] = axes[1].imshow(gr_filt_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[2] = axes[2].imshow(np.abs(gr_float_plot - gr_filt_plot), cmap='gray')
        fig.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=.9, wspace=0.00, hspace=0.00)
        [fig.colorbar(ax_img, ax=ax, orientation='horizontal',pad=0.05, shrink=0.9) for ax_img, ax in zip(ax_imgs, axes)]

    if not mask is None:
        return contrast, contrast_masked
    else:
        return contrast, None

def compute_gaussian_filt_contrast(gr, mask=None, sigma=1.0, kernel_size=0, no_bits=8, debug=False):
    '''
    Convert image to float and compute its RMS local contrast using a gaussian filtered image

    Parameters
    ----------
    gr : MxN unit8 numpy array
        Input grayscale image.
    sigma : float, optional
        Sigma for the gaussian kernel. The default is 1.0.
    kernel_size : int, optional
        Kernel size for gaussian if 0 auto calculate size as round(sigma * 4 * 2 +1).
        The default is 0.
    no_bits : int, optional
        Bit depth of input image. The default is 8.
    debug : bool, optional
        Show computation plots if true. The default is False.

    Returns
    -------
    contrast : float
        Returns the computed gaussian filtered RMS local contrast.
    '''

    if gr.dtype != np.uint8:
        raise ValueError("Provided image gr is not uint8")
    if gr.ndim != 2:
        raise ValueError("Provided image gr is not grayscale")
    if not mask is None and mask.dtype is not np.dtype('bool'):
        raise ValueError("Provided mask in not bool")

    if kernel_size == 0:
        kernel_size = int(np.round(sigma * 4 * 2 + 1))

    if kernel_size % 2 == 0:
        raise ValueError ("Kernel size must be an odd integer")

    levels = 2 ** no_bits - 1
    gr_float = gr.astype(np.float32) / levels

    ignore_border_width = kernel_size // 2

    gr_filt = cv2.GaussianBlur(gr_float, ksize=(kernel_size, kernel_size),
                               sigmaX=sigma, sigmaY=sigma,
                               borderType = cv2.BORDER_ISOLATED)

    gr_float_cropped = gr_float[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    gr_filt_cropped = gr_filt[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    if not mask is None:
        mask_cropped = mask[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
        diff_masked = gr_float_cropped[mask_cropped] - gr_filt_cropped[mask_cropped]
        contrast_masked = np.sum(diff_masked ** 2) / np.sum(mask_cropped)

    contrast = np.sum( (gr_float_cropped - gr_filt_cropped)**2 ) / ( gr_float_cropped.shape[0] * gr_float_cropped.shape[1] )

    if debug:
        if not mask is None:
            gr_float_plot = gr_float * mask
            gr_filt_plot = gr_filt * mask
        else:
            gr_float_plot = gr_float
            gr_filt_plot = gr_filt

        print("Kernel size: {}".format(kernel_size))
        fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
        ax_imgs = np.empty((13),dtype=object)
        axes[0].set_title('Raw')
        axes[1].set_title('Gaussian Filtered sigma={:.2f} kernel_size={}'.format(sigma, kernel_size))
        axes[2].set_title('Diff')
        [axi.set_axis_off() for axi in axes.ravel()]

        ax_imgs[0] = axes[0].imshow(gr_float_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[1] = axes[1].imshow(gr_filt_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[2] = axes[2].imshow(np.abs(gr_float_plot - gr_filt_plot), cmap='gray')
        fig.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=.9, wspace=0.00, hspace=0.00)
        [fig.colorbar(ax_img, ax=ax, orientation='horizontal',pad=0.05, shrink=0.9) for ax_img, ax in zip(ax_imgs, axes)]

    if not mask is None:
        return contrast, contrast_masked
    else:
        return contrast, None

def compute_bilateral_filt_contrast(gr, mask=None, sigmaSpace=1.0, sigmaColor=0.1, kernel_size=0, no_bits=8, debug=False):
    '''
    Convert image to float and compute its RMS local contrast using a bilateral filtered image

    Parameters
    ----------
    gr : MxN unit8 numpy array
        Input grayscale image.
    sigmaSpace : float, optional
        Sigma for the gaussian spatial kernel. The default is 1.0.
    sigmaColor : float, optional
        Sigma for the gaussian intensity kernel. The default is 0.1
    kernel_size : int, optional
        Kernel size for bilateral filter if 0 auto calculate size as round(sigmaSpace * 4 * 2 +1).
        The default is 0.
    no_bits : int, optional
        Bit depth of input image. The default is 8.
    debug : bool, optional
        Show computation plots if true. The default is False.

    Returns
    -------
    contrast : float
        Returns the computed gaussian filtered RMS local contrast.
    '''
    if gr.dtype != np.uint8:
        raise ValueError("Provided image gr is not uint8")
    if gr.ndim != 2:
        raise ValueError("Provided image gr is not grayscale")
    if not mask is None and mask.dtype is not np.dtype('bool'):
        raise ValueError("Provided mask in not bool")


    if kernel_size == 0:
        kernel_size = int(np.round(sigmaSpace * 4 * 2 + 1))

    if kernel_size % 2 == 0:
        raise ValueError ("Kernel size must be an odd integer")

    levels = 2 ** no_bits - 1
    gr_float = gr.astype(np.float32) / levels

    ignore_border_width = kernel_size // 2

    gr_filt = cv2.bilateralFilter(gr_float, kernel_size, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace, borderType=cv2.BORDER_ISOLATED)


    gr_float_cropped = gr_float[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    gr_filt_cropped = gr_filt[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
    
    if not mask is None:
        mask_cropped = mask[ignore_border_width : -ignore_border_width, ignore_border_width : -ignore_border_width]
        diff_masked = gr_float_cropped[mask_cropped] - gr_filt_cropped[mask_cropped]
        contrast_masked = np.sum(diff_masked ** 2) / np.sum(mask_cropped)

    contrast = np.sum( (gr_float_cropped - gr_filt_cropped)**2 ) / ( gr_float_cropped.shape[0] * gr_float_cropped.shape[1] )

    if debug:
        if not mask is None:
            gr_float_plot = gr_float * mask
            gr_filt_plot = gr_filt * mask
        else:
            gr_float_plot = gr_float
            gr_filt_plot = gr_filt

        
        fig, axes = plt.subplots(1,3, sharex=True, sharey=True)
        ax_imgs = np.empty((13),dtype=object)
        axes[0].set_title('Raw')
        axes[1].set_title('Bilateral Filtered space sig={:.2f} color sig={:.2f}'.format(sigmaSpace, sigmaColor))
        axes[2].set_title('Diff')
        [axi.set_axis_off() for axi in axes.ravel()]

        ax_imgs[0] = axes[0].imshow(gr_float_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[1] = axes[1].imshow(gr_filt_plot, cmap='gray',vmin=0, vmax=1)
        ax_imgs[2] = axes[2].imshow(np.abs(gr_float_plot - gr_filt_plot), cmap='gray')
        fig.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=.9, wspace=0.00, hspace=0.00)
        [fig.colorbar(ax_img, ax=ax, orientation='horizontal',pad=0.05, shrink=0.9) for ax_img, ax in zip(ax_imgs, axes)]

    if not mask is None:
        return contrast, contrast_masked
    else:
        return contrast, None
