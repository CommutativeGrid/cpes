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
import plotly.express as px
from operator import itemgetter
import random


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
        neighbours_indices=[tuple(j for j,value in enumerate(vector) if value<2 * self.radius*1.001 and j!=i) for i,vector in enumerate(distance_matrix)]
        neighbours_count=[len(t) for t in neighbours_indices]
        return df.assign(neighbours_count=neighbours_count,neighbours=neighbours_indices)

    def thinning(self,mode,number_removal,from_layer=0,to_layer=1,style="survived",save_path=None,inplace=True,is_removable=None):
        """
        mode: singlet, doublet, triplet
        thinning with paired thinning
        remove adjacent pairs of atoms
        from_layer: the layer to remove atoms from
        to_layer: the layer to remove atoms to
        layer_joined: the layer that the atom is currently in
        """
        assert number_removal >=0
        df=self.df.copy()
        if is_removable is None:
            removable_index_list=list(df.loc[df["layer_joined"]==from_layer].index)
        elif is_removable=="all":
            removable_index_list=list(df.index)
        else:
            if is_removable not in df.columns:
                raise ValueError(f"{is_removable} is not in the dataframe")
            removable_index_list = list(df[pd.concat([df[is_removable], df["layer_joined"]==from_layer],axis=1).all(axis=1)].index)
        if mode == "singlet":
            atoms_removed = random.sample(removable_index_list,number_removal)
        elif mode == "doublet":
            number_removal = number_removal // 2
            nodes_g1 = random.sample(removable_index_list,number_removal)
            nodes_g2=[]
            remove_from_g1 =[]
            for node1 in nodes_g1:
                # node1 is the seed for growing a pair
                # only remove an adjacent atom that is also interior
                candidates=set([_ for _ in df.loc[node1].neighbours if df.loc[_].is_interior])
                if len(candidates)==0: # see if candidates is empty
                    print("No more adjacent interior atom.")
                    remove_from_g1.append(node1)
                    continue
                else:
                    node2=random.choice(list(candidates))
                    nodes_g2.append(node2)
            nodes_g1=list(set(nodes_g1)-set(remove_from_g1))
            atoms_removed=set([*nodes_g1,*nodes_g2])
        elif mode == "triplet":
            number_removal = number_removal // 3
            nodes_g1 = random.sample(removable_index_list,number_removal)
            nodes_g2=[]
            nodes_g3=[]
            remove_from_g1 =[]
            for node1 in nodes_g1:
                # node1 is the seed for growing a triangle
                flag="searching"
                candidates_n2=set([_ for _ in df.loc[node1].neighbours if df.loc[_].is_interior]) # only remove an adjacent atom that is also interior
                if len(candidates_n2)==0: # see if candidates is empty
                    print("No more adjacent interior atom.")
                    remove_from_g1.append(node1)
                    continue
                candidates_n2_shuffled=list(candidates_n2)
                random.shuffle(candidates_n2_shuffled)
                for node2 in candidates_n2_shuffled: # iterate over the shuffled set, no need to remove 
                    candidates_n3=candidates_n2.intersection(set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior]))
                    if len(candidates_n3)==0:
                        continue
                    else:
                        node3=random.choice(list(candidates_n3))
                        nodes_g2.append(node2)
                        nodes_g3.append(node3)
                        flag="found"
                        break
                if flag=="searching":
                    print(f"No interior triangle can be built from point {node1}.")
                    remove_from_g1.append(node1)
                    # seems that can always find a triangle
                # while True:
                #     try:
                #         node2=random.choice(tuple(candidates_n2))
                #         candidates_n3=candidates_n2.intersection(set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior]))
                    
                #     except IndexError: # no more candidate in candidates_n2
                #         print("No more adjacent interior atom.")
                #         break
                #     try:
                #         node3=random.choice(tuple(candidates_n3))
                #         nodes_g2.append(node2)
                #         nodes_g3.append(node3)
                #         break
                #     except IndexError: # no candidate in candidates_n3
                #         candidates_n2.remove(node2)
                #         print("Removing unqualified node2")
                #         continue
            nodes_g1=list(set(nodes_g1)-set(remove_from_g1))
            atoms_removed = set([*nodes_g1,*nodes_g2,*nodes_g3])
        elif mode == "quartet":
            # guaranteed to find a tetrahedron because we use intersections here
            number_removal = number_removal // 4
            nodes_g1 = random.sample(removable_index_list,number_removal)
            nodes_g2=[]
            nodes_g3=[]
            nodes_g4=[]
            remove_from_g1 =[]
            for node1 in nodes_g1:
                # node1 is the seed for growing a tetrahedron
                flag="searching"
                candidates_n2=set([_ for _ in df.loc[node1].neighbours if df.loc[_].is_interior])
                if len(candidates_n2)==0: # see if candidates is empty
                    print("No more adjacent interior atom.")
                    remove_from_g1.append(node1)
                    continue
                candidates_n2_shuffled=list(candidates_n2)
                random.shuffle(candidates_n2_shuffled) # modified inplace
                for node2 in candidates_n2_shuffled: # iterate over the shuffled set, no need to remove
                    candidates_n3=candidates_n2.intersection(set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior]))
                    if len(candidates_n3)==0:
                        continue
                    candidates_n3_shuffled=list(candidates_n3)
                    random.shuffle(candidates_n3_shuffled)
                    for node3 in candidates_n3_shuffled:
                        candidates_n4=candidates_n3.intersection(set([_ for _ in df.loc[node3].neighbours if df.loc[_].is_interior]))
                        if len(candidates_n4)==0:
                            continue
                        else:
                            node4=random.choice(list(candidates_n4))
                            nodes_g2.append(node2)
                            nodes_g3.append(node3)
                            nodes_g4.append(node4)
                            flag="found"
                            break
                    if flag=="found":
                        break
                if flag=="searching":
                    print(f"No interior tetrahedron can be built from point {node1}.")
                    remove_from_g1.append(node1)
                # while True:
                #     try:
                #         node2=random.choice(tuple(candidates_n2))
                #         candidates_n3=candidates_n2.intersection(set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior]))
                #     except IndexError:
                #         print("No more adjacent interior atom.")
                #         break
                #     try:
                #         node3=random.choice(tuple(candidates_n3))
                #         candidates_n4=candidates_n3.intersection(set([_ for _ in df.loc[node3].neighbours if df.loc[_].is_interior]))
                #     except IndexError:
                #         candidates_n2.remove(node2)
                #         print("No more adjacent interior atom.")
                #         break
                #     try:
                #         node4=random.choice(tuple(candidates_n4))
                #         nodes_g2.append(node2)
                #         nodes_g3.append(node3)
                #         nodes_g4.append(node4)
                #         break
                #     except IndexError:
                #         candidates_n2.remove(node2)
                #         print("Removing unqualified node2")
            nodes_g1=list(set(nodes_g1)-set(remove_from_g1))
            atoms_removed = set([*nodes_g1,*nodes_g2,*nodes_g3,*nodes_g4])
        elif "chain" in mode:
            chain_length = int(mode.split('-')[0])
            number_removal = number_removal // chain_length
            chains_found=0
            failed=0
            atoms_removed=set()
            while True:
            #for _ in range(number_removal):
                flag="searching"
                selected=set()
                new_node = random.choice(removable_index_list)
                selected.add(new_node)
                for i in range(chain_length-1): # & takes the intersection of two sets
                    # should we use iloc here?

                    candidates=set([_ for _ in df.loc[new_node].neighbours if df.loc[_].is_interior])\
                        & set(removable_index_list)\
                            - selected
                    if len(candidates)==0:
                        flag="failed"
                        break
                    else:
                        new_node=random.choice(list(candidates))
                        selected.add(new_node)
                if flag=="failed":
                    failed+=1
                else:
                    flag="found"
                    chains_found+=1
                    removable_index_list=list(set(removable_index_list)-selected)
                    atoms_removed = atoms_removed | selected
                if chains_found>=number_removal:
                    break
                elif failed>=number_removal:
                    raise ValueError("No chain can be built from the given set of atoms.")
        elif mode == "quintet":
            # will not find a qunitet in FCC
            # is searching for a tetrahedron
            number_removal = number_removal // 5
            nodes_g1 = random.sample(removable_index_list,number_removal)
            nodes_g2=[]
            nodes_g3=[]
            nodes_g4=[]
            nodes_g5=[]
            remove_from_g1 =[]
            for node1 in nodes_g1:
                candidates_n2=set([_ for _ in df.loc[node1].neighbours if df.loc[_].is_interior])
                # should we use iloc here?
                if len(candidates_n2)==0: # see if candidates is empty
                    print("No more adjacent interior atom.")
                    remove_from_g1.append(node1)
                    continue
                candidates_n2_shuffled=list(candidates_n2)
                random.shuffle(candidates_n2_shuffled)
                flag="searching"
                for node2 in candidates_n2_shuffled: # iterate over the shuffled set, no need to remove
                    candidates_n3=candidates_n2.intersection(set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior]))
                    if len(candidates_n3)==0:
                        continue
                    candidates_n3_shuffled=list(candidates_n3)
                    random.shuffle(candidates_n3_shuffled)
                    for node3 in candidates_n3_shuffled:
                        candidates_n4=candidates_n3.intersection(set([_ for _ in df.loc[node3].neighbours if df.loc[_].is_interior]))
                        if len(candidates_n4)==0:
                            continue
                        else:
                            node4=random.choice(list(candidates_n4))
                            flag="found"
                            break
                    if flag=="found":
                        break # break out of the for loop of node2
                if flag == "searching":
                    print(f"No interior tetrahedron can be built from point {node1}.")
                    continue
                # create the list of the common neighbours of 123,124,134, and 234
                neighbours_1=set([_ for _ in df.loc[node1].neighbours if df.loc[_].is_interior or True])
                neighbours_2=set([_ for _ in df.loc[node2].neighbours if df.loc[_].is_interior or True])
                neighbours_3=set([_ for _ in df.loc[node3].neighbours if df.loc[_].is_interior or True])
                neighbours_4=set([_ for _ in df.loc[node4].neighbours if df.loc[_].is_interior or True])
                print(node1,node2,node3,node4)
                print(f"neighbours_1: {neighbours_1}")
                print(f"neighbours_2: {neighbours_2}")
                print(f"neighbours_3: {neighbours_3}")
                print(f"neighbours_4: {neighbours_4}")
                candidates_n5=neighbours_1 & neighbours_2 & neighbours_3 | neighbours_1 & neighbours_2 & neighbours_4 | neighbours_1 & neighbours_3 & neighbours_4 | neighbours_2 & neighbours_3 & neighbours_4
                candidates_n5 = candidates_n5 - set({node1,node2,node3,node4})
                print(candidates_n5)
                if len(candidates_n5)==0:
                    flag="searching"
                else:
                    node5=random.choice(list(candidates_n5))
                    nodes_g2.append(node2)
                    nodes_g3.append(node3)
                    nodes_g4.append(node4)
                    nodes_g5.append(node5)
                if flag=="searching":
                    remove_from_g1.append(node1)
                    print(f"No interior hexahedron can be built from point {node1}.")
            nodes_g1=list(set(nodes_g1)-set(remove_from_g1))
            atoms_removed = set([*nodes_g1,*nodes_g2,*nodes_g3,*nodes_g4,*nodes_g5])
        else:
            raise NotImplementedError(f"Mode {mode} not implemented.")
        print(f"Mode: {mode}, changing the layer of {len(atoms_removed)} atoms.")
        if not atoms_removed:
            print("No atom is removed.")
            return None
        df.loc[list(atoms_removed),"layer_joined"]=to_layer
        output_df=df.loc[df["layer_joined"]==from_layer]
        if inplace:
            print("Replacing the original lattice layer data with the new one.")
            self.df=output_df
            # return sth to indicate the operation is successful
            return True
        elif not inplace:
            if style == "survived":
                # instead of returning a ClosePacking object, we return a dataframe,
                # because after thinning this is no longer a close packing
                # is this a reasonble argument? 
                # maybe we should return a ClosePacking object with the new dataframe
                return output_df
            elif style == "homcloud":
                labelled_data=[(int(l),*vec) for l,*vec in df[["layer_joined","x","y","z",]].values]
                sorted_result = np.array(sorted(labelled_data, key=itemgetter(0)))
                if save_path is not None:
                    np.savetxt(save_path,sorted_result,fmt=["%d"] + ["%.6f"] * 3,delimiter=" ")
                    #remove the trailing newline
                    with open(save_path) as f_input:
                        data = f_input.read().rstrip('\n')
                    with open(save_path, 'w') as f_output:    
                        f_output.write(data)
                    print(f"File saved @ {save_path} in homcloud format.")
                    return save_path
                else:
                    return sorted_result
            else:
                raise NotImplementedError(f"Style {style} is not implemented.")



    def thinning_obsolete(self, survival_rate=None, number_removal=None, save_path=None, style="survived", inplace=False, is_removable="is_interior"):
        print("Only interior points are involved in the thinning process.")
        return super().thinning(survival_rate=survival_rate, number_removal=number_removal, save_path=save_path, style=style, inplace=inplace, is_removable=is_removable)

    def interiorPoints_count(self):
        """
        return the number of interior points
        """
        return self.df["is_interior"].sum()

    def plot_coordination_number(self,color_column="neighbours_count"):
        """
        plot with coordination number implied
        color_column: neighbours or is_interior
        """
        fig = px.scatter_3d(self.df, x='x', y='y', z='z',
              color=color_column)
        #fig.show()
        return fig



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
        self.df=self.df.assign(is_interior=self.df["neighbours_count"]==12,layer_joined=0)
        

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
        self.df=self.df.assign(is_interior=self.df["neighbours_count"]==12,layer_joined=0)

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