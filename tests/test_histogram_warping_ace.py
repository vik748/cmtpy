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
from cmtpy.histogram_warping_ace import histogram_warping_ace

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
        for gr in self.gr_list:
            gr_warped = histogram_warping_ace(gr, lam = 5, no_bits = 8, plot_histograms=False)
            self.assertEqual(gr_warped.shape, gr.shape, "Return image size incorrect")

    def test_histogram_warping_plots(self):
        """
        Test the warping with plotting
        """
        for gr in self.gr_list:
            gr_warped = histogram_warping_ace(gr, lam = 5, no_bits = 8, plot_histograms=False)
            self.assertEqual(gr_warped.shape, gr.shape, "Return image size incorrect")

if __name__ == '__main__':
    unittest.main()
