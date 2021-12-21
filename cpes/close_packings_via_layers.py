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
import os


# create fcc and hcp by adding layers
# fcc layer A,B,C
# hcp layer A,B
# for details see https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres

# @dataclass
# class layer:
#     name: str #A,B or A,B,C
#     radius: float # default is 0.5
#     nx: int # number of cells in x direction
#     ny: int # number of cells in y direction
#     coords: np.ndarray # coordinates of each atom/center of hexagon in this layer

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

class points_3d:
    """
    Basic class for 3d points
    """
    def __init__(self, data):
        self.data = data

    def plot(self,alpha=0.8,size=400):
        """plot spheres in 3d"""
        #fig=plt.figure(figsize=(10,10),dpi=100)
        fig=plt.figure()
        ax=fig.add_subplot(111,projection='3d')
        color_vector=[self.color_map(*vec) for vec in self.data]
        ax.scatter(self.data[:,0],self.data[:,1],self.data[:,2],alpha=alpha,s=size,c=color_vector)

    def color_map(self,x,y,z)->float:
        """function used for assigning different colors for different types of atoms."""
        return z

    def __len__(self):
        return len(self.data)

    def __repr__(self) -> str:
        return self.data.__repr__()

    def distance_array(self,n=20):
        """
        Compute the distance to the k-th nearest neighbor for all atoms in the point cloud.
        """
        return distance_array(self.data,n)

    def sphere_confine(self,radius):
        return points_3d(sphere_confine(self.data,radius))
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

    def add_perturbtation(self):
        """
        add perturbation to the point cloud
        """
        mean=0
        sigma=1e-3
        flattened=self.data.flatten()
        flattened_perturbed = flattened + np.random.normal(mean,sigma,len(flattened))
        self.data = flattened_perturbed.reshape(self.data.shape)


class close_packing(points_3d):

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
        super().__init__(data=np.empty((0,3), float))

    def thinning(self,survival_rate,save=False,style='homcloud'):
        data = thinning(self.data,survival_rate,style=style)
        if save:
            cwd = os.getcwd()
            file_name=f'{type(self).__name__}_{self.num}_{survival_rate}_thinned.out'
            file_path = os.path.join(cwd,file_name)
            if os.path.isfile(file_path):
                raise FileExistsError(f'File {file_path} already exists.')
            else:
                np.savetxt(file_path,data,fmt=['%d']+['%.6f']*3,delimiter=' ')
                print(f'File saved @ {file_path}')
        else:
            return data

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

    def ratio(self):
        """
        return the packing density
        """
        return atomic_packing_factor(self.data,self.radius)


class face_centered_cubic(close_packing):
    """
    cubic close packing / face centered cubic
    pattern ABCABCACB...
    """
    def __init__(self,num=5,radius=0.5,num_vector='auto',perturbation=True):
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
        self.shift_z = self.translation[2] # used in color_map to modify the color accordingly
        #self.data=np.around(self.data,decimals=7)
        if perturbation is True:
            self.add_perturbtation()


    def color_map(self,x,y,z):
        """function used for assigning different colors for different types of atoms."""
        # use np.around to avoid 0.9999999%1=0.99999 problem (expecting 0).
        coeff=self.multiplier
        return (np.around((z+self.shift_z)/sqrt(6),2))%coeff

class hexagonal_close_packing(close_packing):
    """
    cubic close packing
    """
    def __init__(self,num=5,radius=0.5,num_vector='auto',perturbation=True):
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
        self.shift_z = self.translation[2] # used in color_map to modify the color accordingly
        #self.data=np.around(self.data,decimals=7)
        if perturbation is True:
            self.add_perturbtation()
        
    def color_map(self,x,y,z):
        """function used for assigning different colors for different types of atoms."""
        coeff=self.multiplier
        return (np.around(1.5*(z+self.shift_z)/sqrt(6),2))%coeff
    