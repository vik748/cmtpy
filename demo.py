#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:10:35 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import histogram_warping_ace
import cv2

data_path = os.path.dirname(os.path.relpath(data.__file__))
gr = cv2.imread(os.path.join('test_data','histogram_warping_test_image.png'),
                  cv2.IMREAD_GRAYSCALE)

gr_warped = histogram_warping_ace(gr, lam = 5, no_bits = 8, plot_histograms=False)