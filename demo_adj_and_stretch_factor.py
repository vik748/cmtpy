#!/usr/bin/env pyt
n3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import HistogramWarpingACE
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
    plt.pause(1e-3)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

def analyze_contrast(gr_name, graph_axes, iceberg_slice=np.s_[:,:], set_name=None):
    gr_full = cv2.imread(gr_name, cv2.IMREAD_GRAYSCALE)
    gr = cv2.resize(gr_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

    adjs = np.arange(0,-1.2,-.2)
    sfs = np.arange(-1.0,0.2,.2)
    sfs_g, adj_g = np.meshgrid(sfs, adjs)

    ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=-1.0, stretch_factor=-1.0,
                                  min_stretch_bits=4, downsample_for_kde=True,debug=False, plot_histograms=False)
    v_k, a_k = ace_obj.compute_vk_and_ak(gr)

    warped_images = np.empty(adj_g.shape,dtype=object)

    fig,axes = plt.subplots(*adj_g.shape, sharex=True, sharey=True)
    for axi in axes.ravel():
        axi.get_xaxis().set_ticks ([])
        axi.get_yaxis().set_ticks ([])
        axi.spines['left'].set_visible(False)
        axi.spines['right'].set_visible(False)
        axi.spines['bottom'].set_visible(False)
        axi.spines['top'].set_visible(False)

    for (i,j),adj in np.ndenumerate(adj_g):
        print(i,adj)
        outputs = ace_obj.compute_bk_and_dk(v_k, a_k, adjustment_factor=adj, stretch_factor=sfs_g[i,j])
        warped_images[i,j], Tx = ace_obj.transform_image(*outputs, gr)

        axes[i,j].imshow(warped_images[i,j], cmap='gray', vmin=0, vmax=255)
        #ax.set_title("Adj factor = {:.2f}".format(adj))

    for i, sf in enumerate(sfs):
        axes[-1,i].set_xlabel('Stretch: {:.2f}'.format(sf))


    for j, adj in enumerate(adjs):
        axes[j,0].set_ylabel('Adj: {:.2f}'.format(adj))


    fig.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=.9, wspace=0.00, hspace=0.00)
    fig.suptitle(set_name)
    show_plot()

    '''
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
    '''

if sys.platform == 'darwin':
    data_fold=os.path.expanduser('~/Google Drive/data')
else:
    data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')
gr3_name = os.path.join(data_fold, 'chess_board','GOPR1488.JPG')

gr1_iceberg_slice = np.s_[205:310,:]
gr2_iceberg_slice = np.s_[130:280,:]


fig0, axes0 = plt.subplots(1,1, num=1)
axes0.set_xlabel('Contrast Adjustment Factor')
axes0.set_ylabel('Global Contrast Factor')

#analyze_contrast(gr1_name, axes0, gr1_iceberg_slice, set_name='Lars1')
analyze_contrast(gr2_name, axes0, gr2_iceberg_slice, set_name='Lars2')
analyze_contrast(gr3_name, axes0, gr2_iceberg_slice, set_name='Chessboard')

