#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 10:39:08 2020

@author: vik748
"""
from cmtpy.histogram_warping_ace import HistogramWarpingACE
import cv2
import sys, os

data_fold=os.path.expanduser('~/data')

gr1_name = os.path.join(data_fold,'Lars1_080818','G0287250.JPG')
gr2_name = os.path.join(data_fold,'Lars2_081018','G0029490.JPG')
gr3_name = os.path.join(data_fold, 'chess_board','GOPR1488.JPG')


gr_full = cv2.imread(gr3_name, cv2.IMREAD_GRAYSCALE)
gr = cv2.resize(gr_full, (0,0), fx=1/5, fy=1/5, interpolation=cv2.INTER_AREA)

ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, stretch_factor=-1, adjustment_factor=-1, debug=True, plot_histograms=True)
ace_obj.apply(gr)