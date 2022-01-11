#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:58:28 2021

@author: kasumi
"""
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from hexalattice.hexalattice import create_hex_grid
from scipy.spatial import distance

from .aux_functions import *


class Layer:
    """
    Basic layer class for planar close packing
    radius is 0.5 by default
    """

    def __init__(self, nx, ny, type="A"):
        self.type = type  # A, B or C
        self.radius = 0.5  # default is 0.5
        # Avoid Argument error in hex_grid: 
        # nx, ny and n are expected to be int type, instead of np.int64 or np.int32
        self.nx = int(nx)
        self.ny = int(ny)
        hex_centers, _ = create_hex_grid(nx=self.nx, ny=self.ny, do_plot=False)
        if type == "A":
            self.shift = np.array([0, 0])
        elif type == "B":
            # translation vector for layer of type B
            self.shift = np.array([0.5, sqrt(3) / 6])
        elif type == "C":
            # translation vector for layer of type C
            self.shift = np.array([1, sqrt(3) / 3])

        self.coords = hex_centers + self.shift  # shift the whole layer accordingly


class Points3d:
    """
    Basic class for 3d points
    """

    def __init__(self, data):
        self.data = data

    def plot(self, alpha=0.8, size=400):
        """plot spheres in 3d"""
        # fig=plt.figure(figsize=(10,10),dpi=100)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        color_vector = [self.color_map(*vec) for vec in self.data]
        ax.scatter(
            self.data[:, 0],
            self.data[:, 1],
            self.data[:, 2],
            alpha=alpha,
            s=size,
            c=color_vector,
        )

    def color_map(self, x, y, z) -> float:
        """function used for assigning different colors for different types of atoms."""
        return z

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def distance_array(self, n=20):
        """
        Compute the distance to the k-th nearest neighbor for 
        all atoms in the point cloud.
        """
        return distance_array(self.data, n)

    def sphere_confine(self, radius):
        return Points3d(sphere_confine(self.data, radius))

    @property
    def diameter(self):
        """
        returns the diameter of the point cloud
        """
        if hasattr(self, "_diameter"):
            return self._diameter
        else:
            self._diameter = max(
                distance.cdist(self.data, self.data, "euclidean").flatten()
            )
            return self._diameter

    def add_perturbtation(self):
        """
        add perturbation to the point cloud
        """
        mean = 0
        sigma = 1e-4
        flattened = self.data.flatten()
        flattened_perturbed = flattened + np.random.normal(mean, sigma, len(flattened))
        self.data = flattened_perturbed.reshape(self.data.shape)


class ClosePacking(Points3d):
    def __init__(self, num, radius, num_vector, *args, **kwargs):
        """a class with support methods for close packings

        Args:
            num (int): number of spheres in each direction
            radius (float): radius of the sphere
            num_vector (np.array): will be (num,num,num) if not provided by the user.
        """
        self.num = num
        if num_vector == "auto":
            self.num_vector = np.array([num, num, num], dtype=int)
        else:
            print("Parameter num is ignored. Using num_vector instead.")
            self.num_vector = num_vector
        self.radius = radius
        print(f"Generating close packing with atom radius {self.radius}.")
        # self.diameter = 2*radius # diameter of the sphere, not the point cloud
        self.z_step = sqrt(6) / 3  # tranlsation amount in the vertical direction
        # the default radius is 0.5, so we need to multiply by this to 
        # get the actual radius
        self.multiplier = radius / 0.5
        # will create the default sized layers first
        super().__init__(data=np.empty((0, 3), float))
        self.thinning_history = []

    def thinning(self, survival_rate, save=False, style="homcloud", replace=False):
        """delete points randomly from data.

        Parameters
        ----------
        data : ndarray
            the data to be thinned
        save : bool, optional
            whether to save the thinned data to a file, by default False
        survival_rate : float
            percentage of points to survive
        style : str, optional
            the style of thinning. The default is 'homcloud'.
            'homcloud' : remained points will be labelled 0.
            'survived' : only returns the survived points.

        """
        (data, sorted_result) = thinning(self.data, survival_rate, style=style)
        if replace:
            self.data = data
            self.thinning_history.append(survival_rate)
        if style == "homcloud":
            if save:
                cwd = os.getcwd()
                file_name = (
                    f"{type(self).__name__}_{self.num}_{survival_rate}_thinned.out"
                )
                file_path = os.path.join(cwd, file_name)
                if os.path.isfile(file_path):
                    raise FileExistsError(f"File {file_path} already exists.")
                else:
                    np.savetxt(
                        file_path,
                        sorted_result,
                        fmt=["%d"] + ["%.6f"] * 3,
                        delimiter=" ",
                    )
                    print(f"File saved @ {file_path} in homcloud format.")
            else:
                return sorted_result
        elif style == "survived":
            # sorted_result is None
            if save:
                file_name = (
                    f"{type(self).__name__}_{self.num}_{survival_rate}_thinned.out"
                )
                file_path = os.path.join(cwd, file_name)
                if os.path.isfile(file_path):
                    raise FileExistsError(f"File {file_path} already exists.")
                else:
                    np.savetxt(file_path, data, delimiter=" ")
                    print(f"File saved @ {file_path}.")
            else:
                return Points3d(data)
        else:
            raise NotImplementedError()

    @staticmethod
    def lift(layer, z):
        """
        lift a layer of atoms represented by a 2d numpy array
        """
        return np.array([[x, y, z] for x, y in layer.coords])

    def rdf_plot(self, start=2, end=6, step=0.01, divided_by_2=True):
        return rdf_plot(self.data, start, end, step, divided_by_2)

    def coordination_number(self):
        """
        coordination number
        """
        return coordination_number(self.data, 2 * self.radius)

    def ratio(self):
        """
        return the packing density
        """
        return atomic_packing_factor(self.data, self.radius)


class FaceCenteredCubic(ClosePacking):
    """
    cubic close packing / face centered cubic
    pattern ABCABCACB...
    """

    def __init__(self, num=5, radius=1, num_vector="auto", perturbation=False):
        super().__init__(num=num, radius=radius, num_vector=num_vector)
        nx, ny, nz = self.num_vector
        layer_A = Layer(nx=nx, ny=ny, type="A")
        layer_B = Layer(nx=nx, ny=ny, type="B")
        layer_C = Layer(nx=nx, ny=ny, type="C")
        for i in range(nz):
            if i % 3 == 0:
                self.data = np.append(
                    self.data, self.lift(layer_A, i * self.z_step), axis=0
                )
            elif i % 3 == 1:
                self.data = np.append(
                    self.data, self.lift(layer_B, i * self.z_step), axis=0
                )
            elif i % 3 == 2:
                self.data = np.append(
                    self.data, self.lift(layer_C, i * self.z_step), axis=0
                )
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.shift_z = self.translation[
            2
        ]  # used in color_map to modify the color accordingly
        # self.data=np.around(self.data,decimals=7)
        if perturbation is True:
            print("Adding perturbation to the point cloud.")
            self.add_perturbtation()

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        # use np.around to avoid 0.9999999%1=0.99999 problem (expecting 0).
        coeff = self.multiplier
        return (np.around((z + self.shift_z) / sqrt(6), 2)) % coeff


class HexagonalClosePacking(ClosePacking):
    """
    cubic close packing
    """

    def __init__(self, num=5, radius=1, num_vector="auto", perturbation=False):
        super().__init__(num=num, radius=radius, num_vector=num_vector)
        nx, ny, nz = self.num_vector
        layer_A = Layer(nx=nx, ny=ny, type="A")
        layer_B = Layer(nx=nx, ny=ny, type="B")
        for i in range(nz):
            if i % 2 == 0:
                self.data = np.append(
                    self.data, self.lift(layer_A, i * self.z_step), axis=0
                )
            elif i % 2 == 1:
                self.data = np.append(
                    self.data, self.lift(layer_B, i * self.z_step), axis=0
                )
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.shift_z = self.translation[
            2
        ]  # used in color_map to modify the color accordingly
        # self.data=np.around(self.data,decimals=7)
        if perturbation is True:
            print("Adding perturbation to the point cloud.")
            self.add_perturbtation()

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        coeff = self.multiplier
        return (np.around(1.5 * (z + self.shift_z) / sqrt(6), 2)) % coeff
