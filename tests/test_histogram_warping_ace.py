#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 23:29:59 2020

@author: vik748
"""

import sys, os
if os.path.dirname(os.path.realpath(__file__)) == os.getcwd():
    sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cv2
import unittest
from cmtpy.histogram_warping_ace import HistogramWarpingACE

class TestHistogramWarpingACE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw_images = ['museum_raw.png', 'mountain_raw.jpg', 'sunset_raw.jpg']

        cls.gr_list = []
        for raw_name in raw_images:
            gr_raw_name = os.path.join('test_data', raw_name)
            gr_img = cv2.imread(gr_raw_name, cv2.IMREAD_GRAYSCALE)
            assert gr_img is not None, "Couldn't read image"
            cls.gr_list.append(gr_img)

    def setup(self):
        self.gr_list = cls.gr_list

    def test_histogram_warping(self):
        """
        Test the warping without plotting
        """
        ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=1.0, stretch_factor=1.0,
                                      min_stretch_bits=4, downsample_for_kde=True,debug=False, plot_histograms=False)

        for gr in self.gr_list:
            gr_warped = ace_obj.apply((gr))
            self.assertEqual(gr_warped.shape, gr.shape, "Return image size incorrect")

    def test_histogram_warping_plots(self):
        """
        Test the warping with plotting
        """
        ace_obj = HistogramWarpingACE(no_bits=8, tau=0.01, lam=5, adjustment_factor=1.0, stretch_factor=1.0,
                                      min_stretch_bits=4, downsample_for_kde=True,debug=False, plot_histograms=True)
        for gr in self.gr_list:
            gr_warped = ace_obj.apply(gr)
            self.assertEqual(gr_warped.shape, gr.shape, "Return image size incorrect")

if __name__ == '__main__':
    unittest.main()
