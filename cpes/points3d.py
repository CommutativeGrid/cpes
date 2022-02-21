import numpy as np
import pandas as pd
import plotly.express as px
from .utils import *
from scipy.spatial import distance



class Points3d:
    """
    Basic class for 3d points, with color vector
    """

    def __init__(self, df=pd.DataFrame({'x' : [],'y' : [],'z' : [],'type' : []})):
        self.df = df
        #self.color_vector=pd.Series(df['type'])

    @property
    def xyz(self):
        return self.df.iloc[:, :3].to_numpy(dtype=float) # return xyz coordinates
    
    @property
    def data(self):
        return self.df.iloc[:, :3].to_numpy(dtype=float) # return xyz coordinates

    @xyz.setter
    def xyz(self,xyz_coord):
        self.df.iloc[:,:3]=xyz_coord

    @data.setter
    def data(self,xyz_coord):
        self.df.iloc[:,:3]=xyz_coord

    @property
    def color_vector(self):
        return pd.Series(self.df['type'])

    @color_vector.setter
    def color_vector(self,new_colors):
        self.df.iloc[:,4]=new_colors

    def plot(self):
        df=self.df
        df['c']=[ord(char) for char in self.df['type']]
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color='c')
        fig.show()

    def plot_legacy(self, alpha=0.8, size=400):
        import matplotlib.pyplot as plt
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
        return len(self.df)

    def __repr__(self) -> str:
        """Display head of self.data if length too large"""
        if len(self.data) > 10:
            return str(self.data[:4])[:-1]+"\n...,\n...,\n"+str(self.data[-4:])[1:]
        #return self.data.__repr__()

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