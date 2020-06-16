#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import HistogramWarpingACE
from cmtpy import contrast_measurement as cm
import cv2
import numpy as np
import sys, os
from matplotlib import pyplot as plt

def calculate_best_screen_packing(N, img_resolution = (800,600), screen_resolution = (1920, 1080)):
    screen_x = screen_resolution[0]
    screen_y = screen_resolution[1]
    screen_area = screen_x * screen_y

    img_x = img_resolution[0]
    img_y = img_resolution[1]
    img_aspect = img_x / img_y

    best_dims = (None,None)
    best_eff = 0.0

    for n_rows in range(1,N//2 +1):
        #print(i)
        n_cols = N // n_rows
        if N % n_rows != 0: n_cols = n_cols+1

        #print(n_rows, n_cols)

        # Test by maximising image height
        img_y_scaled = screen_y / n_rows
        img_x_scaled = img_y_scaled * img_aspect
        img_area_scaled = img_x_scaled * img_y_scaled
        eff = img_area_scaled * N / screen_area
        #print(img_x_scaled, img_y_scaled, eff)

        if eff <= 1.0 and eff > best_eff:
            best_eff = eff
            best_dims = (n_rows, n_cols)

        # Test by maximising image width
        img_x_scaled = screen_x / n_cols
        img_y_scaled = img_x_scaled / img_aspect
        img_area_scaled = img_x_scaled * img_y_scaled
        eff = img_area_scaled * N / screen_area
        #print(img_x_scaled, img_y_scaled, eff)
        if eff <= 1.0 and eff > best_eff:
            best_eff = eff
            best_dims = (n_rows, n_cols)

    #print("Best dims:",best_dims,best_eff)
    return best_dims

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

def analyze_contrast(gr_name, graph_axes, iceberg_slice=np.s_[:,:], set_name=None):
    gr_full = cv2.imread(gr_name, cv2.IMREAD_GRAYSCALE)
    gr = cv2.resize(gr_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

    adjs = np.arange(0,-1.05,-.2)
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



if sys.platform == 'darwin':
    data_fold=os.path.expanduser('~/Google Drive/data')
else:
    data_fold=os.path.expanduser('~/data')

data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')
gr3_name = os.path.join(data_fold, 'chess_board','GOPR1488.JPG')

gr1_iceberg_slice = np.s_[205:310,:]
gr2_iceberg_slice = np.s_[130:280,:]


gr_name = gr1_name
gr_slice = gr1_iceberg_slice
gr = read_grimage(gr_name, resize_scale = 1/5)

contrast_estimators = {'Global Contrast Factor': lambda gr: cm.compute_global_contrast_factor(gr),
                       'RMS Contrast': lambda gr: cm.compute_rms_contrast(gr,debug=False),
                       'Local box filt': lambda gr: cm.compute_box_filt_contrast(gr, kernel_size=17, debug=False),
                       'Local gaussian filt': lambda gr: cm.compute_gaussian_filt_contrast(gr, sigma=5.0, debug=False),
                       'Local bilateral filt': lambda gr: cm.compute_bilateral_filt_contrast(gr, sigmaSpace=5.0, sigmaColor=0.05, debug=False)}


ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=-1.0, stretch_factor=-1.0,
                              min_stretch_bits=4, downsample_for_kde=True,debug=False, plot_histograms=False)
v_k, a_k = ace_obj.compute_vk_and_ak(gr)

adjs = np.arange(0,-1.05,-.05)
contrast_estimates=np.zeros((len(adjs),len(contrast_estimators)))
contrast_estimates_slice=np.zeros((len(adjs),len(contrast_estimators)))

warped_images = np.empty(adjs.shape,dtype=object)

for i,adj in enumerate(adjs):
    print(i,adj)
    outputs = ace_obj.compute_bk_and_dk(v_k, a_k, adjustment_factor=adj, stretch_factor=adj )
    warped_images[i], Tx = ace_obj.transform_image(*outputs, gr)
    contrast_estimates[i,:] =np.array([ce(warped_images[i]) for nm, ce in contrast_estimators.items()])
    contrast_estimates_slice[i,:] =np.array([ce(warped_images[i][gr_slice]) for nm, ce in contrast_estimators.items()])

fig_imgs,axes_imgs = plt.subplots(*calculate_best_screen_packing(len(adjs),img_resolution=gr.shape), sharex=True, sharey=True)
fig_imgs.suptitle(gr_name)
[ax.set_axis_off() for ax in axes_imgs.ravel()]
fig_imgs.subplots_adjust(left=0.01, bottom=0.1, right=0.99, top=.9, wspace=0.01, hspace=0.01)


for wimg, ax, adj in zip(warped_images, axes_imgs.ravel(), adjs):
    ax.imshow(wimg, cmap='gray', vmin=0, vmax=255)
    ax.set_title('Adj fact: {:.2f}'.format(adj))

fig_plot,axes_plot = plt.subplots(2,3, sharex=True)
fig_plot.suptitle(gr_name)

for col, col_slice, ax, est in zip(contrast_estimates.T, contrast_estimates_slice.T, axes_plot.ravel(), contrast_estimators.keys()):
    ax.plot(adjs, col, '.', label='Full')
    ax.plot(adjs, col_slice, '.', label='Iceberg only')
    ax.set_title(est)
    ax.set_xlabel("Adjustment/stretch factor")
    ax.set_ylabel("Calculated contrast")
    ax.legend()


    #axes[i,j].imshow(warped_images[i,j], cmap='gray', vmin=0, vmax=255)
    #ax.set_title("Adj factor = {:.2f}".format(adj))



