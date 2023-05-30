import numpy as np
import pandas as pd
import plotly.express as px
from .utils import *
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.colors import ListedColormap
import pyvista
import os


class Points3D:
    """
    Basic class for 3D points, with color vector
    """

    def __init__(self, df=pd.DataFrame({'x' : [],'y' : [],'z' : [],'type' : []})):
        #breakpoint()
        if type(df) is pd.DataFrame:
            self.df=df
        elif type(df) is np.ndarray:
            self.df=pd.DataFrame(df)
        else:
            raise NotImplementedError("Data type not supported")
        # embed into z=0 plane if a 2d point cloud
        if len(self.df.columns)<2:
            self.df['z']=0
            print("Warning: embedding into z=0 plane")

    @property
    def xyz(self):
        """
        Return the xyz coordinates
        """
        return self.data
    
    @property
    def data(self):
        """
        Return the xyz coordinates
        """
        return self.df.iloc[:, :3].to_numpy(dtype=float)

    @xyz.setter
    def xyz(self,xyz_coord):
        """
        Set the xyz coordinates
        """
        self.data=xyz_coord

    @data.setter
    def data(self,xyz_coord):
        """
        Set the xyz coordinates
        """
        self.df.iloc[:,:3]=xyz_coord

    @property
    def type_vector(self):
        """
        Return the type vector (atoms' type)
        """
        return pd.Series(self.df['type'])

    @type_vector.setter
    def type_vector(self,new_colors):
        """
        Set the type vector
        """
        self.df.iloc[:,4]=new_colors

    def random_delete(self,deletion_rate,inplace=False):
        """
        Delete points randomly based on the given deletion_rate (0-1)
        """
        num_remained=int((1-deletion_rate)*(len(self.df)))
        survivors=np.random.choice(range(len(self.df)),num_remained,replace=False)
        if inplace is True:
            self.df=self.df.iloc[survivors,:].copy()
        else:
            return Points3D(self.df.iloc[survivors,:])
    
    
    def normalise(self):
        """
        Rescale the coordinates such that minimum distance becomes 2"""
        dmat=euclidean_distances(self.data)
        inter_dist=set(np.round(dmat.flatten(),6))
        second_smallest=(list(inter_dist))[1]
        self.data=self.data/(second_smallest/2)

    def is_originCentered(self):
        """
        Test if the mass center is at the origin
        TODO Rewrite this function, compare with the whole supercell
        """
        return np.allclose(self.data[center_point_cloud(self.data)], np.zeros(3))
        #return np.linalg.norm(self.data[center_point_cloud(self.data)])<1e-5

    def plot(self):# plot_plotly
        if any('SPYDER' in name for name in os.environ):
            print("Detected Spyder. Using pyvista insteat of plotly.")
            self.plot_pyvista()
        else:
            pass
        df=self.df
        df['c']=[ord(char) for char in self.df['type']]
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                  color='c')
        fig.show()

    def plot_pyvista(self,backend='pythreejs'):
        """
        Plot the point cloud using pyvista
        points are added to the plotter color by color
        otherwise the displayed colors will not all be correct
        """
        #TODO add legend atom type-color
        plotter=pyvista.Plotter()
        sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        self._add_color()
        for t in set(self.df.type):
            typedf=self.df[self.df.type==t]
            xyz=typedf.loc[:,['x','y','z']].to_numpy(dtype=float)
            color=ListedColormap(typedf['color'].to_numpy())
            pdata = pyvista.PolyData(xyz)
            pdata['orig_sphere'] = np.arange(len(xyz))
            pc = pdata.glyph(scale=False, geom=sphere)
            plotter.add_mesh(pc,cmap=color,show_scalar_bar=False)
        plotter.set_background("royalblue", top="aliceblue")
        plotter.show()
        
        # pdata = pyvista.PolyData(self.xyz)
        # pdata['orig_sphere'] = np.arange(len(self.xyz))
        # # create many spheres from the point cloud
        # sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        # pc = pdata.glyph(scale=False, geom=sphere)
        # pc.plot(cmap=self.color_map(),jupyter_backend=backend)
        # pyvista.set_plot_theme("document")
        # print("Colors of boundary points might not be correct due to a potential bug of pyvista.")
        # plotter.add_mesh(pc,cmap=self.color_map())
        # plotter.set_background("royalblue", top="aliceblue")
        # plotter.show()

    @staticmethod
    def _random_color(alpha=1):
        """return a random rgba color vector"""
        if isinstance(alpha,int) or isinstance(alpha,float):
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,alpha)
        else:
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,np.random.uniform())


    def _add_color(self):
        """
        Add a color column to the dataframe
        """
        point_types=list(set(self.type_vector))
        if not hasattr(self,"palette"):
            palette=[]
            for _ in point_types:
                palette.append(self._random_color())
        else:
            palette=self.palette
        assert len(point_types)==len(palette)
        def type2color(type):
            return palette[point_types.index(type)]
        self.df['color']=self.df['type'].apply(type2color)
        #return ListedColormap(self.df['color'].to_numpy())
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
        #return f"Points3D object of length {len(self.df)}:\n{str(self.data[:5])[:-1]}\n...]"
        return f"Points3D object of length {len(self.df)}:\n{self.df[:5]}\n..."
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
        return Points3D(sphere_confine(self.data, radius))

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

    def rdf_plot(self, start=None, end=6, step=0.01, divided_by_2=True):
        if start is None:
            start=self.distance_array(n=3)[1]*0.9
        return rdf_plot(self.data, start, end, step, divided_by_2)

    def add_perturbation(self):
        """
        add perturbation to the point cloud
        """
        mean = 0
        sigma = 1e-4
        print(f"Adding perturbation with mean {mean} and sigma {sigma}.")
        flattened = self.data.flatten()
        flattened_perturbed = flattened + np.random.normal(mean, sigma, len(flattened))
        self.data = flattened_perturbed.reshape(self.data.shape)

    def thinning(self, survival_rate=None, number_removal=None, save_path=None, style="homcloud", inplace=False, is_removable=None):
        """delete points randomly from data.

        Parameters
        ----------
        data : ndarray
            the data to be thinned
        save : bool, optional
            whether to save the thinned data to a file, by default False
        survival_rate : float
            percentage of points to survive
        style : str, optional
            the style of thinning. The default is 'homcloud'.
            'homcloud' : remained points will be labelled 0.
            'survived' : only returns the survived points.

        """
        (df, sorted_result) = thinning_aux(self.df, survival_rate=survival_rate, number_removal=number_removal, style=style, is_removable=is_removable)
        data=df.iloc[:, [0,1,2]].values
        num=len(self)
        if inplace:
            #self.data = data
            self.df = df
            self.thinning_history.append({"survival_rate":survival_rate, "number_removal":number_removal})
        if style == "homcloud":
            if save_path is not None:
                np.savetxt(
                    save_path,
                    sorted_result,
                    fmt=["%d"] + ["%.6f"] * 3,
                    delimiter=" ",
                )
                #remove the trailing newline
                with open(save_path) as f_input:
                    data = f_input.read().rstrip('\n')
                with open(save_path, 'w') as f_output:    
                    f_output.write(data)
                print(f"File saved @ {save_path} in homcloud format.")
                return save_path
            else:
                return sorted_result
        elif style == "survived":
            # sorted_result is None
            if save_path is not None:
                np.savetxt(save_path, data, delimiter=" ")
                #remove the trailing newline
                with open(save_path) as f_input:
                    data = f_input.read().rstrip('\n')
                with open(save_path, 'w') as f_output:    
                    f_output.write(data)
                print(f"File saved @ {save_path}.")
                return save_path
            else:
                return Points3D(df)
        else:
            raise NotImplementedError()