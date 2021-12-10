#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:19:56 2021

@author: kasumi
"""
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from rdfpy import rdf
import matplotlib.pyplot as plt
from random import choice

def distance_array(data,n=20):
    """
    returns the distance between atoms in increasing order
    By default only the smallest 20 values are returned
    """
    dist_array=euclidean_distances(data).flatten()
    result=np.array(list(set(np.around(dist_array,decimals=4))))
    result.sort()
    return result[:n]

def rdf_plot(data,start=2,end=6,step=0.05,divided_by_2=True):
    """
    plot the radial distribution function
    point cloud needs to be centered
    The result is orientation dependent
    Step shall not be too small
    """
    center=data[center_point_cloud(data)]
    print(f"center of the point cloud is {center}")
    print(f"The point cloud shall be origin centered.")
    points=data
    g_r,radii=rdf(points,dr=step)
    number_pts_per_unit = int(1/step)
    index_left=max(0,start*number_pts_per_unit-5)
    index_right=number_pts_per_unit*end+5
    x_coords=radii[index_left:index_right]
    y_coords=g_r[index_left:index_right]
    if divided_by_2:
        x_coords/=2
    plt.plot(x_coords,y_coords)
    return x_coords,y_coords

def coordination_number(data,crictical_value,anchor='random'):
    """
    Return the number of atoms within radius crictical_value of the anchor.
    When crictical_value is the shortest distance between two atoms, this function
    returns the coordination number of the anchor atom.

    coordination number of the anchor atom in the data.
    result may change when using random mode
    """
    if anchor=='random':
        center_atom=[choice(data)]
    else:
        center_atom=[np.array(anchor).flatten()]
    #TODO remove boundary atoms (or atoms close to the boundary)
    print(center_atom)
    dist_vec=euclidean_distances(center_atom,data).flatten()
    count=0
    for i in dist_vec:
        if i<1.001*crictical_value:
            count+=1
    return count-1 # minus itself

def sphere_confine(data,radius):
    """
    confine the data to a sphere with radius
    """
    return np.array([pt for pt in data if np.linalg.norm(pt)<radius])

def plot_3d(data):
    """
    plot 3d
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:,0], data[:,1], data[:,2],alpha=0.8,s=300.)
    plt.show()
    
def center_point_cloud(data):
    """
    return the index of the point closest to the center of the point cloud
    """
    mass_center=np.mean(data,axis=0)
    return np.argmin(euclidean_distances([mass_center],data))
    