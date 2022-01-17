#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:57:24 2021

@author: kasumi

Read the coordinates data downloaded from isaacs
http://isaacs.sourceforge.net/ex.html#Si
"""

import os
from tempfile import NamedTemporaryFile

import numpy as np

from .close_packings_via_layers import ClosePacking
from .utils import TempfileFromUrl


class ExampleDataIsaacs:
    def __init__(self, url, coord_type, mode, offline_fp=None):
        self.url = url
        self.coord_type = coord_type
        if mode == "online":
            with TempfileFromUrl(url) as fp:
                raw_data = fp.readlines()
            if len(raw_data[-1].split()) <= 1:
                raw_data.pop()  # delete the last line (an emoji or empty)

            with NamedTemporaryFile(delete=False) as fp:
                for _ in raw_data:
                    fp.write(_)
                path = fp.name
        elif mode == "offline":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(dir_path, "issacs_data")
            print(f"Load data from file stored in {data_path}")
            path = offline_fp

        self.original = np.loadtxt(
            path, skiprows=2, usecols=(1, 2, 3), unpack=False
        )  # store the original coordinates
        if mode == "online":
            os.remove(path)

    def __len__(self):
        return len(self.original)


class FccAuCart(ExampleDataIsaacs, ClosePacking):
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/au-cart.xyz"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            dir_name, "isaacs_data", "au-cart.xyz"
        )  # the last line is deleted from the downloaded original file
        super().__init__(url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.cartesian = (
            self.original
        )  # the cartesian coordinates is the same as the original coordinates
        self.normalized = (
            2 * self.cartesian / 2.877925
        )  # rescale to make the radius of the atom 1.0
        # notice that the lattice is not guaranteed to be centerd exactly at the origin
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in close_packing can be used on self.data


# Need to check the lattice data below
# online calculator for transformation of coordinates
# https://cci.lbl.gov/cctbx/frac_cart.html
class HcpRuFrac(ExampleDataIsaacs, ClosePacking):
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/ru-frac.xyz"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            dir_name, "isaacs_data", "ru-frac.xyz"
        )  # the last line is deleted from the downloaded original file
        super().__init__(url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.fractional = (
            self.original
        )  # the fractional coordinates is the same as the original coordinates
        # Transofrm the fractional coordinates to cartesian coordinates
        # parameters from the website
        a = 43.29
        b = 43.29
        c = 68.5
        # alpha = np.pi / 2
        # beta = np.pi / 2
        # gamma = 2 * np.pi / 3
        # n2 = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        # n2=0
        # transformation matrix, see
        # https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
        # M  = np.array([[a,0,0],
        #                [b*np.cos(gamma),b*np.sin(gamma),0],
        #      [c*np.cos(beta),c*n2,c*np.sqrt(np.sin(beta)**2-n2**2)]])
        # pre-computed some values to reduce errors
        M = np.array([[a, 0, 0], [b * (-0.5), b * np.sqrt(3) / 2, 0], [0, 0, c]])

        self.cartesian = self.fractional @ M
        self.normalized = (
            2 * self.cartesian / 5.29987615
        )  # rescale to make the radius of the atom 1.0
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in close_packing can be used on self.data


class FccSiFrac(ExampleDataIsaacs, ClosePacking):
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/si-frac.xyz"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            dir_name, "isaacs_data", "si-frac.xyz"
        )  # the last line is deleted from the downloaded original file
        super().__init__(url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.fractional = (
            self.original
        )  # the fractional coordinates is the same as the original coordinates
        # Transofrm the fractional coordinates to cartesian coordinates
        # parameters from the website
        a = 32.52
        b = 32.52
        c = 32.52
        # alpha = np.pi / 2
        # beta = np.pi / 2
        # gamma = np.pi / 2
        M = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        self.cartesian = self.fractional @ M
        self.normalized = (
            2 * self.cartesian / 4.6939
        )  # rescale to make the radius of the atom 1
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in ClosePacking can be used on self.data
