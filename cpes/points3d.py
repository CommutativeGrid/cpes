import numpy as np
import pandas as pd
import plotly.express as px
from .utils import *
from scipy.spatial import distance
from matplotlib.colors import ListedColormap
import pyvista


class Points3d:
    """
    Basic class for 3d points, with color vector
    """

    def __init__(self, df=pd.DataFrame({'x' : [],'y' : [],'z' : [],'type' : []})):
        self.df = df

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

    def origin_centered(self):
        """
        Test if the mass center is at the origin
        """
        return np.allclose(self.data[center_point_cloud(self.data)], np.zeros(3))
        #return np.linalg.norm(self.data[center_point_cloud(self.data)])<1e-5

    def plot(self):# plot_plotly
        df=self.df
        df['c']=[ord(char) for char in self.df['type']]
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color='c')
        fig.show()

    def plot_pyvista(self,backend='pythreejs'):
        pdata = pyvista.PolyData(self.xyz)
        pdata['orig_sphere'] = np.arange(len(self.xyz))
        # create many spheres from the point cloud
        sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        #https://docs.pyvista.org/api/utilities/_autosummary/pyvista.Sphere.html
        pc = pdata.glyph(scale=False, geom=sphere)
        pc.plot(cmap=self.color_map(),jupyter_backend=backend)
        print("Colors of boundary points might not be correct due to a bug of pyvista.")
        # hcp=HCP(9)
        # hcp.df=hcp.df.iloc[242:265]#change 265 to 263,264 to see the difference
        # hcp.plot_pyvista(backend='panel')

    @staticmethod
    def random_color(alpha=1):
        """return a random rgba color vector"""
        if isinstance(alpha,int) or isinstance(alpha,float):
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,alpha)
        else:
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,np.random.uniform())


    def color_map(self):
        """function used for assigning different colors for different types of atoms."""
        point_types=list(set(self.color_vector))
        if not hasattr(self,"palette"):
            for _ in point_types:
                palette.append(self.random_color())
        else:
            palette=self.palette
        assert len(point_types)==len(palette)
        def type2color(type):
            return palette[point_types.index(type)]
        self.df['color']=self.df['type'].apply(type2color)
        return ListedColormap(self.df['color'].to_numpy())
        # breakpoint()
        # newcolors=np.empty((len(self.df),4))
        # for t,c in zip(point_types,palette):
        #     newcolors[self.df.type==t]=c
        # # change to random colors
        # return ListedColormap(newcolors)

    def __len__(self):
        return len(self.df)

    def __repr__(self) -> str:
        """Display head of self.data if length too large"""
        #return f"Points3d object of length {len(self.df)}:\n{str(self.data[:5])[:-1]}\n...]"
        return f"Points3d object of length {len(self.df)}:\n{self.df[:5]}\n..."
        # if len(self.data) > 10:
        #     return str(self.data[:4])[:-1]+"\n...,\n...,\n"+str(self.data[-4:])[1:]
        # #return self.data.__repr__()

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