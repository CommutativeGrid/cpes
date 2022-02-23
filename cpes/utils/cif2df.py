#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 20:07:24 2022

@author: hina
"""
import pandas as pd
from tempfile import NamedTemporaryFile
import subprocess
from crystals import Crystal
import uuid
from .aux_functions import center_point_cloud
from ..points3d import Points3D
import os


def cif2df(filepath:str, supercell_str:str, origin_centered=True, mode="crystal", **kwargs)->pd.DataFrame:
    """chains the following tasks together:
    1.Receive user input about supercell size
    2.Execute the cif2cell command generated based on user input
    3.Save the output as a temp file
    4.Load the temp file, delete extra lines, save as a dataframe and return

    Args:
        filepath (str): path to the cif file
        supercell (str): supercell size
        translation (bool): whether to translate the points to the origin
        mode (str): either "crystal" or "cif2cell"
        **kwargs: other arguments accepted by cif2cell
    """
    if mode=="crystal":
        output_filename=f"{uuid.uuid4().hex}.xyz"
        crystal = Crystal.from_cif(filepath)
        input_supercell_vector=supercell_str.replace('[','').replace(']','').split(',')
        supercell=crystal.supercell(*[int(_) for _ in input_supercell_vector])
        supercell.to_xyz(output_filename)
        df=pd.read_csv(output_filename,skiprows=2,header=None,names=['type','x','y','z'],delim_whitespace=True)
        os.unlink(output_filename)
    elif mode=="cif2cell":
        with NamedTemporaryFile(suffix='.xyz') as f:
            subprocess.call(['cif2cell', filepath, f'--supercell={supercell_str}', 
                            '--program=xyz', f'--outputfile={f.name}'])
            df=pd.read_csv(f.name,skiprows=2,header=None,names=['type','x','y','z'],delim_whitespace=True)
    df=df.reindex(columns=['x','y','z','type'])
    if origin_centered is True:
        # move the center to the origin
        center_index=center_point_cloud(df.iloc[:,0:3])
        translated=df.iloc[:,0:3]-df.iloc[center_index,0:3]
        df.iloc[:,0:3]=translated
    p3d=Points3D(df)
    #p3d.normalise()
    return p3d