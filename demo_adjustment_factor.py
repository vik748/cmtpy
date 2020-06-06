#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import histogram_warping_ace
from cmtpy.global_contrast_factor import compute_global_contrast_factor
from cmtpy.harris_responnse import plot_harris_eig_vals

import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt
import scipy.stats as st

def show_plot(fig=None):
    if fig is None:
        fig = plt.gcf()

    #plt.show()
    plt.pause(1e-9)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

def analyze_contrast(gr_name, graph_axes, iceberg_slice=np.s_[:,:], set_name=None):
    gr_full = cv2.imread(gr_name, cv2.IMREAD_GRAYSCALE)
    gr = cv2.resize(gr_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

    warped_images = np.empty((11),dtype=object)
    adjs = np.arange(-1,1.2,.2)

    for i, adj in enumerate(adjs):
        warped_images[i], _ = histogram_warping_ace(gr, lam = 5, no_bits = 8, tau = 0.01,
                                                    plot_histograms=False, stretch=False, debug=False,
                                                    return_Tx = True, adjustment_factor = adj)

    fig,axes = plt.subplots(2,5, sharex=True, sharey=True)
    [axi.set_axis_off() for axi in axes.ravel()]
    warped_images_for_plot = np.delete(warped_images,5).reshape(2,5)
    adjs_display = np.delete(adjs,5)

    for adj, ax, img in zip(adjs_display, axes.ravel(), warped_images_for_plot.ravel()):
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title("Adj factor = {:.2f}".format(adj))


    fig.subplots_adjust(left=0.05, bottom=0.1, right=0.99, top=.9, wspace=0.01, hspace=0.01)
    fig.suptitle(set_name)
    show_plot()

    full_img_gcfs = np.zeros(warped_images.shape)
    iceberg_slice_gcfs = np.zeros(warped_images.shape)

    for i, wimg in enumerate(warped_images):
        full_img_gcfs[i] = compute_global_contrast_factor(wimg)
        iceberg_slice_gcfs[i] = compute_global_contrast_factor(wimg[iceberg_slice])

    graph_axes.plot(adjs, full_img_gcfs, '.-',label=set_name+" Full Image" )
    graph_axes.plot(adjs, iceberg_slice_gcfs, '.-',label=set_name+" Iceberg Slice" )
    graph_axes.legend()

    fig2,axes2 = plt.subplots(2,5, sharex=True, sharey=True)
    fig2.suptitle(set_name)

    for adj, ax, img in zip(adjs_display, axes2.ravel(), warped_images_for_plot.ravel()):
        plot_harris_eig_vals(img, ax)
        #plot_harris_eig_vals(img, ax)
        ax.set_title("Adj factor = {:.2f}".format(adj))
        ax.set(adjustable='box', aspect='equal')
        ax.set_xlim(ax.set_ylim(0,None))


if sys.platform == 'darwin':
    data_fold=os.path.expanduser('~/Google Drive/data')
else:
    data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')

gr1_iceberg_slice = np.s_[205:310,:]
gr2_iceberg_slice = np.s_[130:280,:]


fig0, axes0 = plt.subplots(1,1, num=1)
axes0.set_xlabel('Contrast Adjustment Factor')
axes0.set_ylabel('Global Contrast Factor')

analyze_contrast(gr1_name, axes0, gr1_iceberg_slice, set_name='Lars1')
analyze_contrast(gr2_name, axes0, gr2_iceberg_slice, set_name='Lars2')

