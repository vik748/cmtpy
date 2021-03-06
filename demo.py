#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import HistogramWarpingACE
import cv2
import time
import numpy as np
import sys, os
from matplotlib import pyplot as plt
import scipy.stats as st

def show_plot(fig=None):
    if fig is None:
        fig = plt.gcf()

    #plt.show()
    plt.pause(0.1)
    fig.canvas.manager.window.activateWindow()
    fig.canvas.manager.window.raise_()

def plot_img_histograms(gr, axes, title=""):
    axes[0].imshow(gr,cmap='gray', vmin=0, vmax=255)
    axes[0].set_title(title)

    x = np.linspace(0,255, 256)
    x_img = gr.flatten()
    gr_kde_full = st.gaussian_kde(x_img,bw_method='silverman')
    f = gr_kde_full(x)

    # Display Histogram and KDE
    axes[1].hist(x_img, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
    axes[1].fill_between(x, f, color='red',alpha=0.4)
    axes[1].set_xlim(0,255)

    # Display cumulative histogram
    axes[2].hist(x_img, bins=x, color='blue', cumulative=True,
                 density=True, alpha=0.4, label='Raw')

def compare_single_img(gr_raw_name, gr_bm_name):
    gr_raw = cv2.imread(gr_raw_name, cv2.IMREAD_GRAYSCALE)
    gr_bm = cv2.imread(gr_bm_name, cv2.IMREAD_GRAYSCALE)

    ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, stretch_factor=1,adjustment_factor=1, debug=True, plot_histograms=True)
    gr_warp,Tx = ace_obj.apply(gr_raw, return_Tx=True)

    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(5, 3)
    axes = np.empty((4,3),dtype=object)
    axes[0,0] = fig.add_subplot(gs[0:2,0])
    axes[1,0] = fig.add_subplot(gs[2,0])
    axes[2,0] = fig.add_subplot(gs[3,0], sharex = axes[1,0])

    axes[0,1] = fig.add_subplot(gs[0:2,1], sharex = axes[0,0], sharey=axes[0,0])
    axes[1,1] = fig.add_subplot(gs[2,1], sharex = axes[1,0], sharey = axes[1,0])
    axes[2,1] = fig.add_subplot(gs[3,1], sharex = axes[1,1], sharey = axes[2,0])
    axes[3,1] = fig.add_subplot(gs[4,1])

    axes[0,2] = fig.add_subplot(gs[0:2,2])
    axes[1,2] = fig.add_subplot(gs[2,2], sharex = axes[1,0], sharey = axes[1,0])
    axes[2,2] = fig.add_subplot(gs[3,2], sharex = axes[1,2], sharey = axes[2,0])
    axes[3,2] = fig.add_subplot(gs[4,2], sharex = axes[3,1], sharey = axes[3,0])


    [axi.set_axis_off() for axi in axes[0,:].ravel()]
    [plt.setp(a.get_xticklabels(), visible=False) for a in axes[1:3,:].ravel()]
    [plt.setp(a.get_yticklabels(), visible=False) for a in axes[1:,1:].ravel()]

    plot_img_histograms(gr_raw, axes[:,0], title="Raw")
    plot_img_histograms(gr_warp, axes[:,1], title="cmtpy")
    plot_img_histograms(gr_bm, axes[:,2], title="Paper")

    axes[3,2].plot(gr_raw.flatten(), gr_bm.flatten(),'.',ms=.5)
    x = np.linspace(0,1, 256)
    axes[3,1].plot(x*255,Tx(x)*255,'.', ms=0.5)
    show_plot(fig)

plt.close('all')

data_path = 'test_data'

raw_images = ['museum_raw.png', 'mountain_raw.jpg', 'sunset_raw.jpg']

for raw_name in raw_images:
    gr_raw_name = os.path.join(data_path, raw_name)
    gr_bm_name = os.path.join(data_path, raw_name.replace('raw','histogram_warped'))

    compare_single_img(gr_raw_name, gr_bm_name)

plt.show()

