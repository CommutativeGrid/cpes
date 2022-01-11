#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:45:24 2021

@author: kasumi
"""
import numpy as np
from matplotlib import pyplot as plt
from .aux_functions import *


class CrystalSystem:
    def __init__():
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def plot(self):
        """plot 3d atomic coordinates"""
        # fig=plt.figure(figsize=(10,10),dpi=100)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        color_vector = [self.color_map(*vec) for vec in self.data]
        ax.scatter(
            self.data[:, 0],
            self.data[:, 1],
            self.data[:, 2],
            alpha=0.8,
            s=400.0,
            c=color_vector,
        )

    def color_map(self, x, y, z):
        raise NotImplementedError

    def distance_array(self, n=20):
        return distance_array(self.data, n)

    def rdf_plot(self, start=2, end=6, step=0.01, divided_by_2=True):
        return rdf_plot(self.data, start, end, step, divided_by_2)


class SimpleCubic(CrystalSystem):
    """class for generating atomic coordinates of a supercell of simple cubic lattice
    """

    def __init__(self, num_cells_axis=10, side_l=1.0):
        self._unit_grid = []  # atomic coordinates stored in a list
        self.side_l = side_l  # side length of one conventional cell
        self.length = num_cells_axis + 1  # variable used in the loop
        for x in range(self.length):
            for y in range(self.length):
                for z in range(self.length):
                    self._unit_grid.append(
                        [x, y, z]
                    )  # append atomic coordinates at each integer valued points
        self.data = (
            np.array(self._unit_grid) * self.side_l
        )  # convert list to numpy array

    def __len__(self):
        return len(self.data)

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        return x % self.side_l + y % self.side_l + z % self.side_l


class BodyCenteredCubic(SimpleCubic):
    """bcc class"""

    def __init__(self, num_cells_axis=10, side_l=1.0):
        self.side_l = side_l
        super().__init__(num_cells_axis, side_l=self.side_l)
        self.side_l_half = self.side_l / 2.0
        for x in range(0, self.length - 1):
            for y in range(0, self.length - 1):
                for z in range(0, self.length - 1):
                    self._unit_grid.append(
                        [x + 0.5, y + 0.5, z + 0.5]
                    )  # add an atom at the body center of each conventional cell
        self.data = (
            np.array(self._unit_grid) * self.side_l
        )  # convert list to numpy array
