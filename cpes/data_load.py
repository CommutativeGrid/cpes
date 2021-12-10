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

# TODO: load fractional coordinates of hcp.