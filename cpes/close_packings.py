#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 15:45:24 2021

@author: kasumi
"""
from math import sqrt

import numpy as np

from .aux_functions import *
from .cubic_crystal_systems import CrystalSystem, SimpleCubic


class FaceCenteredCubic(SimpleCubic):
    """fcc class
    cubic close packing
    """

    def __init__(self, num_cells_axis=10, radius="sqrt(2)/4"):
        """
        a = side length
        radius = sqrt(2)/4 * a
        """
        if radius == "sqrt(2)/4":
            self.side_l = 1.0
        else:
            self.side_l = 4 * radius / sqrt(2)
        self.radius = radius
        super().__init__(num_cells_axis=num_cells_axis, side_l=self.side_l)
        for x in range(0, self.length - 1):
            for y in range(0, self.length - 1):
                for z in range(0, self.length - 1):
                    self._unit_grid.append([x, y + 0.5, z + 0.5])
                    self._unit_grid.append([x + 0.5, y, z + 0.5])
                    self._unit_grid.append([x + 0.5, y + 0.5, z])
        # add atoms on three boundary surfaces
        for x in range(self.length - 1, self.length):
            for y in range(0, self.length - 1):
                for z in range(0, self.length - 1):
                    self._unit_grid.append([x, y + 0.5, z + 0.5])
        for y in range(self.length - 1, self.length):
            for x in range(0, self.length - 1):
                for z in range(0, self.length - 1):
                    self._unit_grid.append([x + 0.5, y, z + 0.5])
        for z in range(self.length - 1, self.length):
            for x in range(0, self.length - 1):
                for y in range(0, self.length - 1):
                    self._unit_grid.append([x + 0.5, y + 0.5, z])
        self.data = np.array(self._unit_grid) * self.side_l
        self.data = (
            self.data - self.data[center_point_cloud(self.data)]
        )  # centralize the point cloud
        self.data = np.around(self.data, decimals=7)

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        coeff = 3 * self.side_l
        return (x + y + z) % coeff


class HexagonalClosePacking(CrystalSystem):
    """hcp class
    hexagonal close packing
    coordinates from wikipedia
    https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres

    multiplies of sqrt can cause errors. e.g.:
        a=sqrt(2e8)
        b=1e4*sqrt(2)
        print(f"{a:.20f}\n{b:.20f}")
    variable `a` will be closer to the real value.

    see the following page for visualization
    http://lampx.tugraz.at/~hadley/ss1/crystalstructure/structures/hcp/hcp.php
    """

    def __init__(self, num_cells_axis=10, radius=sqrt(2) / 4):
        self.radius = radius
        self.length = num_cells_axis + 2
        self._unit_grid = []
        # sqrt3=sqrt(3)
        # coeff_z=2*sqrt(6)/3.0
        for x in range(0, self.length):
            for y in range(0, self.length):
                for z in range(0, self.length):
                    temp = y + (z % 2) / 3.0
                    self._unit_grid.append(
                        [
                            2 * x + (y + z) % 2,
                            1 + sqrt(
                                3 * (temp * temp)
                            ),
                            # include the parameter within the square root
                            # to increase precision
                            1 + 2 * sqrt(6 * z * z) / 3,
                        ]
                    )
        self.data = np.array(self._unit_grid) * self.radius
        self.data = (
            self.data - self.data[center_point_cloud(self.data)]
        )  # centralize the point cloud
        self.data = np.around(self.data, decimals=7)

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        coeff = sqrt(6) * 2 * self.radius
        return z % coeff


if __name__ == "__main__":
    a = FaceCenteredCubic(num_cells_axis=10, radius=1)
    # a.plot()
    b = HexagonalClosePacking(num_cells_axis=10, radius=1)
    # b.plot()
    a.rdf_plot()
