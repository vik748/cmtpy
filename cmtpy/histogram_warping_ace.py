#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
histogra_warping

@author: vik748
"""
import numpy as np
import sys, os
import scipy.stats as st
from scipy.signal import argrelmin
from scipy import interpolate as interp
from scipy import integrate
try:
    import cv2
    opencv_available = True
except ModuleNotFoundError as e:
    print("Opencv Not avialable using 0 order hold for resizing: {}".format(e))
    opencv_available = False

def gen_F_inverse(F,x_d, delta = 1e-4):
    '''
    Given a cumulative F and gray values x_d
    '''
    zero_indices = np.where(F<delta)[0]
    last_zero = zero_indices[-1] if len(zero_indices)>0 else 0

    one_indices = np.where(F>(1-delta))[0]
    first_one = one_indices[0] if len(one_indices)>0 else len(F)

    F_trunc = np.copy(F[last_zero : first_one])
    F_trunc[0] = 0
    F_trunc[-1] = 1
    x_d_trunc = np.copy(x_d[last_zero : first_one])

    #for f,x in zip(F_trunc, x_d_trunc):
    #    print(x,f)

    F_interp = interp.interp1d(F_trunc, x_d_trunc)
    return F_interp

def get_Transform(a,b,d,x):
    '''
    a = array of segment mid points
    b = location of the adjusted mid points
    d = slops
    '''

    assert len(a) == len(b) == len(d)

    x_map =  np.array([])
    T_x =  np.array([])
    for k in range(1,len(a)):
        r_k = ( b[k] - b[k-1] ) / ( a[k] - a[k-1] )
        x_in =  x[np.logical_and(x > a[k-1], x < a[k])]
        t = ( x_in - a[k-1] ) /  (a[k] - a[k-1] )
        T = b[k-1] + \
            ( ( r_k * t**2 + d[k-1]*(1-t)*t ) * ( b[k] - b[k-1] ) /
              ( r_k + ( d[k] + d[k-1] - 2*r_k ) * (1-t) * t ) )

        x_map = np.concatenate((x_map, x_in))
        T_x = np.concatenate((T_x, T))

    return interp.interp1d(x_map, T_x,
                           bounds_error = False,
                           fill_value = ( np.min(T_x), np.max(T_x) ) )

def calc_scale_factor(orig_size):
    orig_size_arr = np.array(orig_size)
    scale_found = False
    scale_factor = 0
    while not scale_found:
        scale_factor += 1
        new_size = orig_size_arr / scale_factor
        if np.max(new_size) < 1000:
            remainders = new_size % scale_factor
            scale_found = remainders[0] == 0 and remainders[1] == 0
    return int(scale_factor)

def histogram_warping_ace(gr, lam = 5.0, no_bits = 8, plot_histograms=False,
                          tau = 0.01, stretch = False, downsample_for_kde = True, debug = False):

    if gr.ndim > 2:
        raise ValueError("Number of dims > 2, image might not be grayscale")

    if downsample_for_kde and np.max(gr.shape) > 1000:
        sc_fac = calc_scale_factor(gr.shape)
        if debug: print("Scale factor = ", sc_fac)
        if opencv_available:
            gr_sc = cv2.resize(gr, (0,0), fx=1/sc_fac, fy=1/sc_fac, interpolation=cv2.INTER_AREA)
        else:
            gr_sc = gr[::sc_fac,::sc_fac]
    else:
        gr_sc = gr

    no_bits = 8
    no_gray_levels = 2 ** no_bits

    x = np.linspace(0,1, no_gray_levels)
    x_img = gr.flatten() / (no_gray_levels - 1)
    x_img_sc = gr_sc.flatten() / (no_gray_levels - 1)

    #h = 0.7816774 * st.iqr(x_img) * ( len(x_img) ** (-1/7) )
    #h = 0.7816774 * ( Finv_interp(0.75) - Finv_interp(0.25) ) * ( len(x_img) ** (-1/7) )
    x_kde_full = st.gaussian_kde(x_img_sc, bw_method='silverman')
    x_kde = x_kde_full(x)

    f = x_kde
    F = np.cumsum(f)
    F = np.concatenate((np.array([0]), integrate.cumtrapz(f, x)))

    f_interp = interp.interp1d(x, f)
    F_interp = interp.interp1d(x, F)
    Finv_interp = gen_F_inverse(F,x, delta = 1e-4)

    valleys = x[argrelmin(f)[0]]
    #v_k = np.concatenate( (np.array([0]), valleys,np.array([1]) ) )
    v_k = np.concatenate( ( Finv_interp([0]), valleys, Finv_interp([1]) ) )
    a_k = (v_k[0:-1] + v_k[1:])/2


    vk = v_k[1:]
    vk1 = v_k[0:-1]
    b_k = ( (F_interp(vk)  - F_interp(a_k)) * vk1 +
            (F_interp(a_k) - F_interp(vk1)) * vk  ) / \
          (  F_interp(vk)  - F_interp(vk1) )

    a_k_full = np.concatenate( ( Finv_interp([0]), a_k, Finv_interp([1]) ) )
    b_k_full = np.concatenate( ( np.array([0]), b_k, np.array([1]) ) )
    if debug:
        print("a_k_full_unscaled: ",a_k_full)
        print("b_k_full_unscaled: ",b_k_full)

    if stretch:
        # Stretch and scale ranges
        a_k_full_scaled = np.copy(a_k_full)
        a_k_full_scaled[1] = Finv_interp(tau)
        a_k_full_scaled[-2] = Finv_interp(1-tau)
        #a_k_full_scaled[2:-2] =  a_k_full_scaled[1] + (a_k_full[2:-2] - a_k_full[1]) / \
        #                                              (a_k_full[-2] - a_k_full[1]) * \
        #                                              (a_k_full_scaled[-2] - a_k_full_scaled[1])

        b_k_full_scaled = np.copy(b_k_full)
        b_k_full_scaled[1] = tau
        b_k_full_scaled[-2] = 1-tau
        b_k_full_scaled[2:-2] =  b_k_full_scaled[1] + (b_k_full[2:-2] - b_k_full[1]) / \
                                                      (b_k_full[-2] - b_k_full[1]) * \
                                                      (b_k_full_scaled[-2] - b_k_full_scaled[1])

        a_k_full_unscaled = a_k_full
        b_k_full_unscaled = b_k_full
        a_k_full = a_k_full_scaled
        b_k_full = b_k_full_scaled

        if debug:
            print("a_k_full_scaled: ", a_k_full)
            print("b_k_full_scaled: ", b_k_full)

    # Calculate dk
    a_k = a_k_full[1:-1]
    a_k_plus_1 = a_k_full[2:]
    a_k_minus_1 = a_k_full[0:-2]

    a_k_plus = Finv_interp((F_interp(a_k) + F_interp(a_k_plus_1))/2)    #.astype(int)
    a_k_minus = Finv_interp((F_interp(a_k) + F_interp(a_k_minus_1))/2)  #.astype(int)

    b_k = b_k_full[1:-1]
    b_k_plus_1 = b_k_full[2:]
    b_k_minus_1 = b_k_full[0:-2]

    b_k_plus =  ( b_k + b_k_plus_1  ) / 2
    b_k_minus = ( b_k + b_k_minus_1 ) / 2

    exp_denom = F_interp(a_k_plus_1) - F_interp(a_k_minus_1)
    if debug: print("exp_denom: ",exp_denom)

    first_term = ( (b_k - b_k_minus) / (a_k - a_k_minus) ) ** \
                 ( ( F_interp(a_k) - F_interp(a_k_minus_1) ) / exp_denom )

    second_term = ( (b_k_plus - b_k) / (a_k_plus - a_k) ) ** \
                  ( ( F_interp(a_k_plus_1) - F_interp(a_k) ) / exp_denom )

    d_k = first_term * second_term

    b_0_plus = ( b_k_full[0] + b_k_full[1] ) / 2
    a_0_plus = Finv_interp( ( F_interp(a_k_full[0]) + F_interp(a_k_full[1]) ) /2 )
    d_0 = b_0_plus / a_0_plus

    b_K_plus_1_minus = ( b_k_full[-1] + b_k_full[-2] ) / 2
    a_K_plus_1_minus = Finv_interp( ( F_interp(a_k_full[-1]) + F_interp(a_k_full[-2]) )/2 )
    d_K_plus_1 = ( 1 - b_K_plus_1_minus ) / ( 1 - a_K_plus_1_minus )

    d_k_full = np.concatenate( (np.array([d_0]), d_k, np.array([d_K_plus_1]) ) )
    if debug: print("d_k_full before threshold: ",d_k_full)

    d_k_full[d_k_full < 1/lam] = 1/lam
    d_k_full[d_k_full > lam]
    if debug: print("d_k_full after threshold: ",d_k_full)

    T_x_interp = get_Transform(a_k_full, b_k_full, d_k_full, x)

    x_img_adj = T_x_interp(x_img)
    gr_warp = np.round(x_img_adj * (no_gray_levels - 1)).astype(np.uint8).reshape(gr.shape)


    if plot_histograms:

        def x_coord_lines(x_arr, y_arr,ax, labels=None, *args, **kwargs ):
            for i,(x, y) in enumerate(zip(x_arr, y_arr)):
                ax.plot([x,x], [0,y],'o--',*args, **kwargs)
                if labels is not None:
                    ax.annotate(labels+"{}".format(i), (x,y/2),*args, **kwargs)

        def y_coord_lines(x_arr, y_arr, ax, labels=None, *args, **kwargs ):
            for i,(x, y) in enumerate(zip(x_arr, y_arr)):
                ax.plot([0,x], [y,y],'o--',*args, **kwargs)
                if labels is not None:
                    ax.annotate(labels+"{}".format(i), (x/2,y),*args, **kwargs)

        try:
            from matplotlib import pyplot as plt
        except ImportError as e:
            raise SystemExit('Error: {}.\nUnable to import matplotlib, \
                              cannot display histograms'.format(e))

        if downsample_for_kde and np.max(gr_warp.shape) > 1000:
            sc_fac = calc_scale_factor(gr_warp.shape)
            if debug: print("Scale factor = ", sc_fac)
            if opencv_available:
                gr_warp_sc = cv2.resize(gr_warp, (0,0), fx=1/sc_fac, fy=1/sc_fac, interpolation=cv2.INTER_AREA)
            else:
                gr_warp_sc = gr_warp[::sc_fac,::sc_fac]
        else:
            gr_warp_sc = gr_warp


        x_img_adj_sc = gr_warp_sc.flatten() / (no_gray_levels - 1)
        x_adj_kde_full = st.gaussian_kde(x_img_adj_sc, bw_method=x_kde_full.silverman_factor())
        x__adj_kde = x_adj_kde_full(x)

        f_adj = x__adj_kde
        F_adj = np.cumsum(f_adj)
        F_adj = np.concatenate((np.array([0]), integrate.cumtrapz(f_adj, x)))

        # Setup figure
        fig = plt.figure(constrained_layout=True)
        gs = fig.add_gridspec(4, 2)
        axes = np.empty((3,2),dtype=object)
        axes[0,0] = fig.add_subplot(gs[0:2,0])
        axes[1,0] = fig.add_subplot(gs[2,0])
        axes[2,0] = fig.add_subplot(gs[3,0], sharex = axes[1,0])

        axes[0,1] = fig.add_subplot(gs[0:2,1])
        axes[1,1] = fig.add_subplot(gs[2,1], sharey = axes[1,0])
        axes[2,1] = fig.add_subplot(gs[3,1], sharex = axes[1,1], sharey = axes[2,0])

        [axi.set_axis_off() for axi in axes[0,:].ravel()]

        # Display original Image
        axes[0,0].imshow(gr,cmap='gray', vmin=0, vmax=255)
        axes[0,0].set_title("RAW")

        # Display original Histogram and KDE
        axes[1,0].hist(x_img, bins=x, color='blue', density=True, alpha=0.4, label='Raw')
        axes[1,0].fill_between(x, f, color='red',alpha=0.4)
        axes[1,0].set_xlim(0,1)

        # Display cumulative histogram and cumulative KDE
        axes[2,0].hist(x_img, bins=x, color='blue', cumulative=True,
                       density=True, alpha=0.4, label='Raw')
        axes[2,0].fill_between(x, F, color='red',alpha=0.4)

        # Display valleys v_k, mid-points a_k and target mid-pts b_k on histogram and Cumulative histogram
        x_coord_lines(v_k, f_interp(v_k), ax=axes[1,0], labels='v_', color='y')
        x_coord_lines(a_k, f_interp(a_k), ax=axes[1,0], labels='a_', color='g')
        x_coord_lines(b_k, f_interp(a_k), ax=axes[1,0], labels='b_', color='b')

        x_coord_lines(v_k, F_interp(v_k), ax=axes[2,0], labels='v_', color='y')
        y_coord_lines(v_k, F_interp(v_k), ax=axes[2,0], labels='Fv_', color='y')
        x_coord_lines(a_k, F_interp(a_k), ax=axes[2,0], labels='a_', color='g')
        y_coord_lines(a_k, F_interp(a_k), ax=axes[2,0], labels='Fa_', color='g')
        x_coord_lines(b_k, F_interp(a_k), ax=axes[2,0], labels='b_', color='b')

        # Display histogram warped image
        axes[0,1].imshow(gr_warp,cmap='gray', vmin=0, vmax=255)
        axes[0,1].set_title("Histogram Warped")

        # Display warped Histogram and KDE
        axes[1,1].hist(x_img_adj, bins=x, color='blue', density=True, alpha=0.4, label='Warped')
        axes[1,1].fill_between(x, f_adj, color='red',alpha=0.4)
        axes[1,1].set_xlim(0,1)

        # Display warped image cumulative histogram and cumulative KDE
        axes[2,1].hist(x_img_adj, bins=x, color='blue', cumulative=True,
                       density=True, alpha=0.4, label='Raw')
        axes[2,1].fill_between(x, F_adj, color='red',alpha=0.4)

        # Display valleys v_k, mid-points a_k and target mid-pts b_k on histogram and Cumulative histogram
        x_coord_lines(v_k, f_interp(v_k), ax=axes[1,1], labels='v_', color='y')
        x_coord_lines(a_k, f_interp(a_k), ax=axes[1,1], labels='a_', color='g')
        x_coord_lines(b_k, f_interp(a_k), ax=axes[1,1], labels='b_', color='b')

        x_coord_lines(v_k, F_interp(v_k), ax=axes[2,1], labels='v_', color='y')
        y_coord_lines(v_k, F_interp(v_k), ax=axes[2,1], labels='Fv_', color='y')
        x_coord_lines(a_k, F_interp(a_k), ax=axes[2,1], labels='a_', color='g')
        y_coord_lines(a_k, F_interp(a_k), ax=axes[2,1], labels='Fa_', color='g')
        x_coord_lines(b_k, F_interp(a_k), ax=axes[2,1], labels='b_',color='b')

        plt.figure()
        plt.plot(x, T_x_interp(x),'.')

    return gr_warp
