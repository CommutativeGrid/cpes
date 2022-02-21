#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:58:28 2021

@author: kasumi
"""
import os
from math import sqrt
import numpy as np
import pandas as pd
from hexalattice.hexalattice import create_hex_grid


from .utils import *
from .points3d import Points3d


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
        self.type_vector = [type]*len(hex_centers)
        self.coords = hex_centers + self.shift  # shift the whole layer accordingly
        self.coords = np.array([[x,y,0.0] for x,y in self.coords])#embed the layer in a 3d space

        
    
    @property
    def df(self):
        return pd.DataFrame({"x": self.coords[:, 0], "y": self.coords[:, 1], 
                            "z": self.coords[:, 2], "type": self.type_vector})

    def lift(self, raise_z):
        """
        lift a layer of atoms represented by a 2d numpy array
        """
        self.coords = np.array([[x, y, raise_z] for x, y, _ in self.coords])
    


class ClosePacking(Points3d):
    def __init__(self, num, radius, num_vector, *args, **kwargs):
        """a class with support methods for close packings

        Args:
            num (int): number of spheres in each direction
            radius (float): radius of the sphere
            num_vector (np.array): will be (num,num,num) if not provided by the user.
        """
        super().__init__(*args, **kwargs)
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
        self.thinning_history = []

    def thinning(self, survival_rate=None, number_removal=None, save=False, style="homcloud", inplace=False):
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
        (df, sorted_result) = thinning(self.df, survival_rate=survival_rate, number_removal=number_removal, style=style)
        data=df.iloc[:, [0,1,2]].values
        if inplace:
            self.data = data
            self.df = df
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
                return Points3d(df)
        else:
            raise NotImplementedError()

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
                layer_A.lift(i * self.z_step)
                self.df=self.df.append(layer_A.df)
            elif i % 3 == 1:
                layer_B.lift(i * self.z_step)
                self.df=self.df.append(layer_B.df)
            elif i % 3 == 2:
                layer_C.lift(i * self.z_step)
                self.df=self.df.append(layer_C.df)
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.shift_z = self.translation[2]  # used in color_map to modify the color accordingly
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
                layer_A.lift(i * self.z_step)
                self.df=self.df.append(layer_A.df)
            elif i % 2 == 1:
                layer_B.lift(i * self.z_step)
                self.df=self.df.append(layer_B.df)
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.shift_z = self.translation[2]  # used in color_map to modify the color accordingly
        # self.data=np.around(self.data,decimals=7)
        self.df = pd.DataFrame({'x':self.data[:,0],'y':self.data[:,1],'z':self.data[:,2], 'type':self.color_vector},columns=['x','y','z','type'])
        if perturbation is True:
            print("Adding perturbation to the point cloud.")
            self.add_perturbtation()

    def color_map(self, x, y, z):
        """function used for assigning different colors for different types of atoms."""
        coeff = self.multiplier
        return (np.around(1.5 * (z + self.shift_z) / sqrt(6), 2)) % coeff
