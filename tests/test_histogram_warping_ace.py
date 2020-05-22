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
        cls.gr = cv2.imread(os.path.join('test_data','histogram_warping_test_image.png'),
                            cv2.IMREAD_GRAYSCALE)
        assert cls.gr is not None, "Couldn't read image"

    def setup(self):
        self.gr = cls.gr

    def test_histogram_warping(self):
        """
        Test the warping without plotting
        """
        gr_warped = histogram_warping_ace(self.gr, lam = 5, no_bits = 8, plot_histograms=False)
        self.assertEqual(gr_warped.shape, self.gr.shape, "Return image size incorrect")

    def test_histogram_warping_plots(self):
        """
        Test the zernike detector max feature response
        """
        gr_warped = histogram_warping_ace(self.gr, lam = 5, no_bits = 8, plot_histograms=True)
        self.assertTrue(1, "Unable to plot histograms")

if __name__ == '__main__':
    unittest.main()