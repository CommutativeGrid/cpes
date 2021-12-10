#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 15:58:28 2021

@author: kasumi
"""
from hexalattice.hexalattice import create_hex_grid
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from scipy.spatial import distance
import matplotlib.pyplot as plt
from .aux_functions import *


# create fcc and hcp by adding layers
# fcc layer A,B,C
# hcp layer A,B
# for details see https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres

@dataclass
class layer:
    name: str #A,B or A,B,C
    radius: float # default is 0.5
    nx: int # number of cells in x direction
    ny: int # number of cells in y direction
    coords: np.ndarray # coordinates of each atom/center of hexagon in this layer

class layer:
    """
    Basic layer class for planar close packing
    radius is 0.5 by default
    """
    def __init__(self, nx, ny, type='A'):
        self.type = type # A, B or C
        self.radius = 0.5 # default is 0.5
        self.nx = int(nx) # Avoid Argument error in hex_grid: nx, ny and n are expected to be int type, instead of np.int64 or np.int32
        self.ny = int(ny)
        hex_centers, _ = create_hex_grid(nx=self.nx,ny=self.ny,do_plot=False)
        if type == 'A':
            self.shift = np.array([0,0])
        elif type == 'B':
            self.shift = np.array([0.5,sqrt(3)/6]) # translation vector for layer of type B
        elif type == 'C':
            self.shift = np.array([1,sqrt(3)/3]) # translation vector for layer of type C

        self.coords = hex_centers+self.shift # shift the whole layer accordingly

class close_packing:

    def __init__(self,num,radius,num_vector,*args,**kwargs):
        """a class with support methods for close packings

        Args:
            num (int): number of spheres in each direction
            radius (float): radius of the sphere
            num_vector (np.array): will be (num,num,num) if not provided by the user.
        """
        self.num = num
        if num_vector == 'auto':
            self.num_vector = np.array([num,num,num],dtype=int)
        else:
            print('Parameter num is ignored. Using num_vector instead.')
            self.num_vector = num_vector
        self.radius = radius
        #self.diameter = 2*radius # diameter of the sphere, not the point cloud
        self.z_step = sqrt(6)/3 # tranlsation amount in the vertical direction
        self.multiplier = radius/0.5 # the default radius is 0.5, so we need to multiply by this to get the actual radius
        #will create the default sized layers first
        self.data = np.empty((0,3), float)

    def plot(self):
        """plot spheres in 3d"""
        #fig=plt.figure(figsize=(10,10),dpi=100)
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        color_vector=[self.color_map(*vec) for vec in self.data]
        ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2],alpha=0.8,s=400.,c=color_vector)

    def color_map(self,x,y,z):
        """function used for assigning different colors for different types of atoms."""
        raise NotImplementedError

    def __len__(self):
        return len(self.data)
    
    def distance_array(self,n=20):
        return distance_array(self.data,n)

    @property
    def diameter(self):
        """
        returns the diameter of the point cloud
        """
        if hasattr(self,'_diameter'):
            return self._diameter
        else:
            self._diameter = max(distance.cdist(self.data,self.data,'euclidean').flatten())
            return self._diameter

    @staticmethod
    def lift(layer,z):
        """
        lift a layer of atoms represented by a 2d numpy array
        """
        return np.array([[x,y,z] for x,y in layer.coords])
    def rdf_plot(self,start=2,end=6,step=0.01,divided_by_2=True):
        return rdf_plot(self.data,start,end,step,divided_by_2)

    def coordination_number(self):
        """
        coordination number
        """
        return coordination_number(self.data,2*self.radius)

    @staticmethod
    def bounding_box(points):
        """
        helper function for computing the packing density
        """
        x,y,z = zip(*points)
        return [(min(x), min(y), min(z)), (max(x), max(y), max(z))]

    def ratio(self):
        """
        return the packing density
        """
        a,b=self.bounding_box(self.data)
        total=sqrt(3)*distance.euclidean(a,b)**3/9. #volume computed from the body diagonal
        #volume of the bounding box to be optimized.
        occupied=4*pi/3*len(self) # volume of the occupied region
        return occupied/total

class face_centered_cubic(close_packing):
    """
    cubic close packing / face centered cubic
    pattern ABCABCACB...
    """
    def __init__(self,num=5,radius=0.5,num_vector='auto'):
        super().__init__(num=num,radius=radius,num_vector=num_vector)
        nx,ny,nz=self.num_vector
        layer_A = layer(nx=nx,ny=ny,type='A')
        layer_B = layer(nx=nx,ny=ny,type='B')
        layer_C = layer(nx=nx,ny=ny,type='C')
        for i in range(nz):
            if i % 3 == 0:
                self.data = np.append(self.data,self.lift(layer_A,i*self.z_step),axis=0)
            elif i % 3 == 1:
                self.data = np.append(self.data,self.lift(layer_B,i*self.z_step),axis=0)
            elif i % 3 == 2:
                self.data = np.append(self.data,self.lift(layer_C,i*self.z_step),axis=0)
        self.data*=self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation #centralize the point cloud
        self.shift_z = self.translation[2]
        self.data=np.around(self.data,decimals=7)


    def color_map(self,x,y,z):
        """function used for assigning different colors for different types of atoms."""
        # use np.around to avoid 0.9999999%1=0.99999 problem (expecting 0).
        coeff=self.multiplier
        return (np.around((z+self.shift_z)/sqrt(6),2))%coeff

class hexagonal_close_packing(close_packing):
    """
    cubic close packing
    """
    def __init__(self,num=5,radius=0.5,num_vector='auto'):
        super().__init__(num=num,radius=radius,num_vector=num_vector)
        nx,ny,nz=self.num_vector
        layer_A = layer(nx=nx,ny=ny,type='A')
        layer_B = layer(nx=nx,ny=ny,type='B')
        for i in range(nz):
            if i % 2 == 0:
                self.data = np.append(self.data,self.lift(layer_A,i*self.z_step),axis=0)
            elif i % 2 == 1:
                self.data = np.append(self.data,self.lift(layer_B,i*self.z_step),axis=0)
        self.data*=self.multiplier
        self.translation = self.data[center_point_cloud(self.data)]
        self.data = self.data - self.translation #centralize the point cloud
        self.shift_z = self.translation[2]
        self.data=np.around(self.data,decimals=7)
        
    def color_map(self,x,y,z):
        """function used for assigning different colors for different types of atoms."""
        coeff=self.multiplier
        return (np.around(1.5*(z+self.shift_z)/sqrt(6),2))%coeff
    



if __name__ == '__main__':
    fcc=face_centered_cubic(17,radius=1)
    hcp=hexagonal_close_packing(17,radius=1)
    fcc.rdf_plot()
    
# =============================================================================
#     a=layer(10,10,'A')
#     b=layer(10,10,'B')
#     c=layer(10,10,'C')
#     
# =============================================================================

