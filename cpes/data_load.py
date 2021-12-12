#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 16:57:24 2021

@author: kasumi

Read the coordinates data downloaded from isaacs
http://isaacs.sourceforge.net/ex.html#Si
"""

import numpy as np
import os


dir_name=os.path.dirname(os.path.abspath(__file__))
path_fcc=os.path.join(dir_name, "isaacs_data", "au-cart.xyz") # the last line is deleted from the downloaded original file
fcc_au=np.loadtxt(path_fcc,skiprows=2,usecols=(1,2,3),unpack=False)
fcc_au=2*fcc_au/2.877925 # rescale to make the radius of the atom 1.0


# Need to check the lattice data below
# online calculator for transformation of coordinates
# https://cci.lbl.gov/cctbx/frac_cart.html
# # TODO: load fractional coordinates of hcp.
path_hcp_frac=os.path.join(dir_name, "isaacs_data", "ru-frac.xyz") # the last line is deleted from the downloaded original file
hcp_frac_ru=np.loadtxt(path_hcp_frac,skiprows=2,usecols=(1,2,3),unpack=False)
# parameters from the website
a=43.29
b=43.29
c=68.5
alpha=np.pi/2
beta=np.pi/2
gamma=2*np.pi/3
n2 = (np.cos(alpha)-np.cos(beta)*np.cos(gamma))/np.sin(gamma)
#transformation matrix, see https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
# M  = np.array([[a,0,0],
#                [b*np.cos(gamma),b*np.sin(gamma),0], 
#      [c*np.cos(beta),c*n2,c*np.sqrt(np.sin(beta)**2-n2**2)]])
# pre-compute some values to reduce errors
# n2=0
M  = np.array([[a,0,0],
               [b*(-0.5),b*np.sqrt(3)/2,0], 
     [0,0,c]])

hcp_ru=hcp_frac_ru@M
# #hcp_ru = hcp_ru -np.array([ 2.70561959, -1.5620777 ,  2.140625  ])
# hcp_ru=2*hcp_ru/5.29987615


path_si_frac=os.path.join(dir_name, "isaacs_data", "si-frac.xyz")
si_frac=np.loadtxt(path_si_frac,skiprows=2,usecols=(1,2,3),unpack=False)
l=32.52
M  = np.array([[l,0,0],
               [0,l,0], 
     [0,0,l]])
si_cart=si_frac@M
#si_cart=2*si_cart/4.6939