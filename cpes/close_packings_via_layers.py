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
from matplotlib.colors import ListedColormap
from sklearn.metrics.pairwise import euclidean_distances


from .utils import *
from .points3d import Points3D


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
    


class ClosePacking(Points3D):
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

    def neighbours_counting(self,df):
        """
        Compute the number of neighbours for each point
        and add that to the dataframe
        """
        # distance matrix
        distance_matrix = euclidean_distances(df[["x","y","z"]])
        # count the number of neighbours, minus one to remove itself
        num_neighbours=[len([value for value in vector if value<2 * self.radius*1.001])-1 for vector in distance_matrix]
        return df.assign(neighbours=num_neighbours)
        #TODO
        # render colors according to the number of neighbours
        # then decide the value for thinning

    def thinning(self, survival_rate=None, number_removal=None, save_path=None, style="survived", inplace=False, is_removable="is_interior"):
        print("Only interior points are involved in the thinning process.")
        return super().thinning(survival_rate=survival_rate, number_removal=number_removal, save_path=save_path, style=style, inplace=inplace, is_removable=is_removable)

    def interiorPoints_count(self):
        """
        return the number of interior points
        """
        return len(self.df.loc[self.df["is_interior"] == True])


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
                self.df=pd.concat([self.df,layer_A.df])
                #self.df=self.df.append(layer_A.df)
            elif i % 3 == 1:
                layer_B.lift(i * self.z_step)
                self.df=pd.concat([self.df,layer_B.df])
                #self.df=self.df.append(layer_B.df)
            elif i % 3 == 2:
                layer_C.lift(i * self.z_step)
                self.df=pd.concat([self.df,layer_C.df])
                #self.df=self.df.append(layer_C.df)
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.set_palette()
        self.df.reset_index(drop=True,inplace=True)
        if perturbation is True:
            print("Adding perturbation to the point cloud.")
            self.add_perturbation()
        self.df=self.neighbours_counting(self.df)
        self.df=self.df.assign(is_interior=self.df["neighbours"]==12)
        

    def set_palette(self):
        """function used for assigning different colors for different types of atoms."""
        red = np.array([1, 0, 0, 1])
        yellow = np.array([255/256, 247/256, 0/256, 1])
        blue = np.array([12/256, 238/256, 246/256, 1])
        self.palette=[red,yellow,blue]
        # newcolors=np.empty((len(self.df),4))
        # newcolors[self.df.type=='A']=red
        # newcolors[self.df.type=='B']=yellow
        # newcolors[self.df.type=='C']=blue
        # return ListedColormap(newcolors)





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
                self.df=pd.concat([self.df,layer_A.df])
                #self.df=self.df.append(layer_A.df)
            elif i % 2 == 1:
                layer_B.lift(i * self.z_step)
                self.df=pd.concat([self.df,layer_B.df])
                #self.df=self.df.append(layer_B.df)
        self.data *= self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation  # centralize the point cloud
        self.set_palette()
        #self.df = pd.DataFrame({'x':self.data[:,0],'y':self.data[:,1],'z':self.data[:,2], 'type':self.color_vector},columns=['x','y','z','type'])
        self.df.reset_index(drop=True,inplace=True)
        if perturbation is True:
            print("Adding perturbation to the point cloud.")
            self.add_perturbation()
        self.df=self.neighbours_counting(self.df)
        self.df=self.df.assign(is_interior=self.df["neighbours"]==12)

    def set_palette(self):
        """function used for assigning different colors for different types of atoms."""
        red = np.array([1, 0, 0, 1])
        yellow = np.array([255/256, 247/256, 0/256, 1])
        blue = np.array([12/256, 238/256, 246/256, 1])
        self.palette=[red,yellow]

    # def color_map(self):
    #     """function used for assigning different colors for different types of atoms."""
    #     blue = np.array([12/256, 238/256, 246/256, 1])
    #     yellow = np.array([255/256, 247/256, 0/256, 1])
    #     red = np.array([1, 0, 0, 1])
    #     newcolors=np.empty((len(self.df),4))
    #     newcolors[self.df.type=='A']=red
    #     newcolors[self.df.type=='B']=yellow
    #     #newcolors[self.df.type=='C']=blue
    #     return ListedColormap(newcolors)