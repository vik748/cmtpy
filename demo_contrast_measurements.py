#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy import contrast_measurement as cm
import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt

def read_grimage(img_name, resize_scale = None, normalize=False, image_depth=8):
    '''
    Read image from file, convert to grayscale and resize if required
    Parameters
    ----------
    img_name : String
        Filename
    resize_scale : float, optional
        float scale factor for image < 1 downsamples and > 1 upsamples. The default is None.
    normalize : bool, optional
        Return normalized float image. The default is False.
    image_depth : int, optional
        Bit depth of image being read. The default is 8.

    Raises
    ------
    FileNotFoundError
        Raisees FileNotFound Error if unable to read image

    Returns
    -------
    gr : MxN uint8 or flat32 numpy array
        grayscale image
    '''
    img = cv2.imread(img_name, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError ("Could not read image from: {}".format(img_name))
    gr_full = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    if not resize_scale is None:
        gr = cv2.resize(gr_full, (0,0), fx=resize_scale, fy=resize_scale, interpolation=cv2.INTER_AREA)
    else:
        gr = gr_full

    if normalize:
        levels = 2 ** image_depth - 1
        gr = np.divide(gr, levels, dtype=np.float32)

    return gr


def show_plot(fig=None):
    if fig is None:
        fig = plt.gcf()

    #plt.show()
    plt.pause(1e-3)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

if sys.platform == 'darwin':
    data_fold=os.path.expanduser('~/Google Drive/data')
else:
    data_fold=os.path.expanduser('~/data')

data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')
gr3_name = os.path.join(data_fold, 'chess_board','GOPR1488.JPG')
gr4_name = os.path.join('test_data', 'PSNR-example-base.png')
gr5_name = os.path.join('test_data', 'PSNR-example-comp-90.jpg')
gr6_name = os.path.join('test_data', 'PSNR-example-comp-30.jpg')
gr7_name = os.path.join('test_data', 'PSNR-example-comp-10.jpg')


gr = read_grimage(gr2_name, resize_scale = 1/5)

cm.compute_global_contrast_factor(gr)
cm.compute_rms_contrast(gr,debug=True)
cm.compute_box_filt_contrast(gr, kernel_size=17, debug=True)
cm.compute_gaussian_filt_contrast(gr, sigma=5.0, debug=True)
cm.compute_bilateral_filt_contrast(gr, sigmaSpace=5.0, sigmaColor=0.05, debug=True)

gr4 = read_grimage(gr4_name)
gr5 = read_grimage(gr5_name)
gr6 = read_grimage(gr6_name)
gr7 = read_grimage(gr7_name)


