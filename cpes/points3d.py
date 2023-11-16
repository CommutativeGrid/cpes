import numpy as np
import pandas as pd
import plotly.express as px
from .utils import center_point_cloud, distance_array, rdf, thinning_aux, sphere_confine
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from matplotlib.colors import ListedColormap
import pyvista
import os


class Points3D:
    """
    Manages a collection of 3D points with associated types for color categorization.
    """

    def __init__(self, df=pd.DataFrame({'x' : [],'y' : [],'z' : [],'type' : []})):
        """
        Initializes the Points3D object with 3D point data.

        Parameters
        ----------
        df : pandas.DataFrame or numpy.ndarray, optional
            The data containing 'x', 'y', 'z' coordinates and 'type'. An ndarray will be converted to DataFrame.

        Raises
        ------
        NotImplementedError
            If input is not a DataFrame or ndarray.

        Notes
        -----
        If DataFrame has less than 3 columns, points are assumed to be in the z=0 plane.
        """
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
        Gets the xyz coordinates of the points.

        Returns
        -------
        numpy.ndarray
            An array containing the xyz coordinates of the points.
        """
        return self.data
    
    @property
    def data(self):
        """
        Retrieves the xyz coordinates from the DataFrame as a NumPy array.

        Returns
        -------
        numpy.ndarray
            A NumPy array of shape (n, 3), where 'n' is the number of points,
            containing the xyz coordinates.
        """

        return self.df.iloc[:, :3].to_numpy(dtype=float)

    @xyz.setter
    def xyz(self,xyz_coord):
        """
        Sets the xyz coordinates of the points.

        Parameters
        ----------
        xyz_coord : numpy.ndarray
            An array containing the xyz coordinates to set for the points.
        """
        self.data=xyz_coord

    @data.setter
    def data(self,xyz_coord):
        """
        Updates the xyz coordinates in the DataFrame using a NumPy array.

        Parameters
        ----------
        xyz_coord : numpy.ndarray
            A NumPy array of shape (n, 3), where 'n' is the number of points,
            containing the new xyz coordinates to update.
        """
        self.df.iloc[:,:3]=xyz_coord

    @property
    def type_vector(self):
        """
        Gets the type vector representing the category or color of each point.

        Returns
        -------
        pandas.Series
            A series containing the types of each point.
        """
        return pd.Series(self.df['type'])

    @type_vector.setter
    def type_vector(self,new_colors):
        """
        Sets the type vector representing the category or color of each point.

        Parameters
        ----------
        new_colors : array-like
            An array or series representing the new types to set for each point.
        """
        self.df.iloc[:,4]=new_colors

    def random_delete(self,deletion_rate,inplace=False):
        """
        Randomly deletes points based on the specified deletion rate.

        Parameters
        ----------
        deletion_rate : float
            The proportion of points to delete, between 0 and 1.
        inplace : bool, optional
            If True, deletion is done in place and modifies the original data. 
            Otherwise, a new instance with deleted points is returned.

        Returns
        -------
        Points3D or None
            A new Points3D instance with points removed if inplace is False, 
            or None if inplace is True.
        """
        num_remained=int((1-deletion_rate)*(len(self.df)))
        survivors=np.random.choice(range(len(self.df)),num_remained,replace=False)
        if inplace is True:
            self.df=self.df.iloc[survivors,:].copy()
        else:
            return Points3D(self.df.iloc[survivors,:])
    
    
    def normalise(self):
        """
        Rescales the coordinates so that the minimum distance between points becomes 2.
        """ 
        dmat=euclidean_distances(self.data)
        inter_dist=set(np.round(dmat.flatten(),6))
        second_smallest=(list(inter_dist))[1]
        self.data=self.data/(second_smallest/2)

    def is_originCentered(self):
        """
        Checks if the mass center of the point cloud is at the origin.

        Returns
        -------
        bool
            True if the mass center is close to the origin, False otherwise.
        """
        return np.allclose(self.data[center_point_cloud(self.data)], np.zeros(3))
        #return np.linalg.norm(self.data[center_point_cloud(self.data)])<1e-5

    def plot(self):
        """
        Plots the point cloud using Plotly or PyVista, depending on the environment.

        The method detects the running environment and chooses the appropriate plotting library.
        In Spyder, it uses PyVista, while in other environments, it uses Plotly.
        """
        if any('SPYDER' in name for name in os.environ):
            print("Spyder detected. Switching to PyVista for plotting.")
            self.plot_pyvista()
        else:
            df = self.df
            df['c'] = [ord(char) for char in self.df['type']]
            fig = px.scatter_3d(df, x='x', y='y', z='z', color='c', hover_name=df.index)
            fig.show()


    def plot_pyvista_h(self, backend='pythreejs', highlight:list=None):
        """
        Plots the point cloud using PyVista with highlighted points.

        Parameters
        ----------
        backend : str, optional
            The backend used for rendering the plot. Defaults to 'pythreejs'.
        highlight : list, optional
            A list of indices of points to highlight. If None, no points are highlighted.
        """
        #colors https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Color.name.html
        plotter=pyvista.Plotter()
        sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        self._add_color()
        
        for t in set(self.df.type):
            typedf_highlight=self.df.loc[highlight][self.df.type==t]
            typedf_dimmed=self.df.loc[~self.df.index.isin(highlight)][self.df.type==t]
            # import pdb; pdb.set_trace()
            def add_mesh(typedf,opacity=1):
                xyz=typedf.loc[:,['x','y','z']].to_numpy(dtype=float)
                color=ListedColormap(typedf['color'].to_numpy())
                pdata = pyvista.PolyData(xyz)
                pdata['orig_sphere'] = np.arange(len(xyz))
                pc = pdata.glyph(scale=False, geom=sphere)
                plotter.add_mesh(pc, cmap=color, show_scalar_bar=False, opacity=opacity)
            # add mesh if not empty
            if len(typedf_highlight)>0:
                add_mesh(typedf_highlight,opacity=1)
            if len(typedf_dimmed)>0:
                add_mesh(typedf_dimmed,opacity=0.1)

        # Draw lines between specified points
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3), (1, 4), (2, 4), (3, 4)]
        for pair in pairs:
            point1 = self.df.loc[highlight[pair[0]], ['x', 'y', 'z']].values
            point2 = self.df.loc[highlight[pair[1]], ['x', 'y', 'z']].values
            # import pdb; pdb.set_trace()
            line = pyvista.Line(list(point1), list(point2))
            plotter.add_mesh(line, color='lightgray', line_width=14)  # Adjust color and line_width as needed

        plotter.set_background("skyblue", top="aliceblue")
        # plotter.set_background("aliceblue", top="royalblue")
        # plotter.show()
        return plotter
    
    def plot_pyvista_dim2(self, backend='pythreejs', highlight:list=None):
        """
        This is a specific function for plotting a figure in the paper,
        not for general use.
        """
        #colors https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Color.name.html
        plotter=pyvista.Plotter()
        sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        self._add_color()
        
        for t in set(self.df.type):
            typedf_highlight=self.df.loc[highlight][self.df.type==t]
            typedf_dimmed=self.df.loc[~self.df.index.isin(highlight)][self.df.type==t]
            # import pdb; pdb.set_trace()
            def add_mesh(typedf,opacity=1):
                xyz=typedf.loc[:,['x','y','z']].to_numpy(dtype=float)
                color=ListedColormap(typedf['color'].to_numpy())
                pdata = pyvista.PolyData(xyz)
                pdata['orig_sphere'] = np.arange(len(xyz))
                pc = pdata.glyph(scale=False, geom=sphere)
                plotter.add_mesh(pc, cmap=color, show_scalar_bar=False, opacity=opacity)
            # add mesh if not empty
            if len(typedf_highlight)>0:
                add_mesh(typedf_highlight,opacity=1)
            if len(typedf_dimmed)>0:
                add_mesh(typedf_dimmed,opacity=0.1)

        # Draw lines between specified points
        pairs = [(0, 1), (0, 2), (1, 2), (0, 5), (1, 4), (2, 5), (2, 4), (4, 5), (0,3), (1,3), (3,5), (3,4)]
        for pair in pairs:
            point1 = self.df.loc[highlight[pair[0]], ['x', 'y', 'z']].values
            point2 = self.df.loc[highlight[pair[1]], ['x', 'y', 'z']].values
            # import pdb; pdb.set_trace()
            line = pyvista.Line(list(point1), list(point2))
            plotter.add_mesh(line, color='cornflowerblue', line_width=10)  # Adjust color and line_width as needed
        
        # draw semi-transparent triangles
        mesh_triplets =[(0,1,2),(0,3,5),(0,2,5),(0,1,3),(1,2,4),(2,4,5),(1,3,4),(3,4,5)]
        for triplet in mesh_triplets:
            point1 = self.df.loc[highlight[triplet[0]], ['x', 'y', 'z']].values
            point2 = self.df.loc[highlight[triplet[1]], ['x', 'y', 'z']].values
            point3 = self.df.loc[highlight[triplet[2]], ['x', 'y', 'z']].values
            triangle = pyvista.Triangle([list(point1), list(point2), list(point3)])
            plotter.add_mesh(triangle, color='red',opacity=0.2,line_width=0)

        plotter.set_background("aliceblue", top="aliceblue")
        # plotter.set_background("aliceblue", top="royalblue")
        # plotter.show()
        return plotter
    
    def plot_pyvista_tetrahedron(self, backend='pythreejs', highlight:list=None):
        """
        This is a specific function for plotting a figure in the paper,
        not for general use.
        """
        #colors https://docs.pyvista.org/version/stable/api/utilities/_autosummary/pyvista.Color.name.html
        plotter=pyvista.Plotter()
        sphere = pyvista.Sphere(radius=0.2, phi_resolution=30, theta_resolution=30)
        self._add_color()
        
        for t in set(self.df.type):
            typedf_highlight=self.df.loc[highlight][self.df.type==t]
            typedf_dimmed=self.df.loc[~self.df.index.isin(highlight)][self.df.type==t]
            # import pdb; pdb.set_trace()
            def add_mesh(typedf,opacity=1):
                xyz=typedf.loc[:,['x','y','z']].to_numpy(dtype=float)
                color=ListedColormap(typedf['color'].to_numpy())
                pdata = pyvista.PolyData(xyz)
                pdata['orig_sphere'] = np.arange(len(xyz))
                pc = pdata.glyph(scale=False, geom=sphere)
                plotter.add_mesh(pc, cmap=color, show_scalar_bar=False, opacity=opacity)
            # add mesh if not empty
            if len(typedf_highlight)>0:
                add_mesh(typedf_highlight,opacity=1)
            if len(typedf_dimmed)>0:
                add_mesh(typedf_dimmed,opacity=0.1)

        # Draw lines between specified points
        pairs = [(0, 1), (0, 2), (1, 2), (0,3),(1,3),(2,3)]
        for pair in pairs:
            point1 = self.df.loc[highlight[pair[0]], ['x', 'y', 'z']].values
            point2 = self.df.loc[highlight[pair[1]], ['x', 'y', 'z']].values
            # import pdb; pdb.set_trace()
            line = pyvista.Line(list(point1), list(point2))
            plotter.add_mesh(line, color='cornflowerblue', line_width=10)  # Adjust color and line_width as needed
        
        # draw semi-transparent triangles
        mesh_triplets =[(0,1,2),(0,1,3),(0,2,3),(1,2,3)]
        for triplet in mesh_triplets:
            point1 = self.df.loc[highlight[triplet[0]], ['x', 'y', 'z']].values
            point2 = self.df.loc[highlight[triplet[1]], ['x', 'y', 'z']].values
            point3 = self.df.loc[highlight[triplet[2]], ['x', 'y', 'z']].values
            triangle = pyvista.Triangle([list(point1), list(point2), list(point3)])
            plotter.add_mesh(triangle, color='red',opacity=0.2,line_width=0)

        plotter.set_background("aliceblue", top="aliceblue")
        # plotter.set_background("aliceblue", top="royalblue")
        # plotter.show()
        return plotter


    def plot_pyvista(self,backend='pythreejs'):
        """
        Plots the point cloud using PyVista with the specified backend.

        Parameters
        ----------
        backend : str, optional
            The backend used for rendering the plot, default is 'pythreejs'.
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
        """
        Generates a random RGBA color vector.

        Parameters
        ----------
        alpha : int or float, optional
            The alpha (transparency) value of the color. If not specified, a random value is chosen.
            If specified, it should be between 0 (transparent) and 1 (opaque).

        Returns
        -------
        numpy.ndarray
            An RGBA color vector with random RGB values and the specified or random alpha value.
        """
        if isinstance(alpha,int) or isinstance(alpha,float):
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,alpha)
        else:
            rgb=np.random.choice(range(256), size=3)/255
            return np.append(rgb,np.random.uniform())


    def _add_color(self):
        """
        Internally used to add color information to the DataFrame based on point types.
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
        """
        Overrides the default length method to return the count of 3D points in the DataFrame.
        """
        return len(self.df)

    def __repr__(self) -> str:
        """
        Provides a string representation of the Points3D object, 
        display head of self.data if length too large,
        typically including the number of points.
        """
        #return f"Points3D object of length {len(self.df)}:\n{str(self.data[:5])[:-1]}\n...]"
        return f"Points3D object of length {len(self.df)}:\n{self.df[:5]}\n..."
        # if len(self.data) > 10:
        #     return str(self.data[:4])[:-1]+"\n...,\n...,\n"+str(self.data[-4:])[1:]
        # #return self.data.__repr__()

    def distance_array(self, n=20):
        """
        Computes distances to the k-th nearest neighbor for all points in the cloud.

        Parameters
        ----------
        n : int, optional
            The order of the nearest neighbor to which the distance is calculated, default is 20.

        Returns
        -------
        numpy.ndarray
            An array containing the distances to the n-th nearest neighbor for each point.
        """
        return distance_array(self.data, n)

    def sphere_confine(self, radius):
        """
        Confines the points within a sphere of the given radius.

        Parameters
        ----------
        radius : float
            The radius of the sphere within which the points are to be confined.

        Returns
        -------
        Points3D
            A new Points3D instance with points confined within the specified sphere.
        """
        return Points3D(sphere_confine(self.data, radius))

    @property
    def diameter(self):
        """
        Calculates and returns the maximum distance between any two points in the dataset.
        """
        if hasattr(self, "_diameter"):
            return self._diameter
        else:
            self._diameter = max(
                distance.cdist(self.data, self.data, "euclidean").flatten()
            )
            return self._diameter

    def rdf_plot(self, start=None, end=6, step=0.01, divided_by_2=True):
        """
        Plots the radial distribution function (RDF) for the points.

        Parameters
        ----------
        start : float or None, optional
            The starting distance for calculating RDF. If None, it's set to 90% of the third nearest neighbor distance.
        end : float, optional
            The ending distance for the RDF plot, default is 6.
        step : float, optional
            The step size for distance increment in RDF calculation, default is 0.01.
        divided_by_2 : bool, optional
            If True, the RDF values are halved, default is True.

        Returns
        -------
        matplotlib.figure.Figure
            A plot of the radial distribution function.
        """
        if start is None:
            start=self.distance_array(n=3)[1]*0.9
        return rdf(self.data, start, end, step, divided_by_2)

    def add_perturbation(self):
        """
        Adds random perturbation to the points' coordinates.

        This method introduces a small, normally distributed random perturbation to each coordinate
        of the points, with a mean of 0 and a standard deviation of 1e-4.

        Notes
        -----
        The perturbation is applied in place, modifying the original point coordinates.
        """
        mean = 0
        sigma = 1e-4
        print(f"Adding perturbation with mean {mean} and sigma {sigma}.")
        flattened = self.data.flatten()
        flattened_perturbed = flattened + np.random.normal(mean, sigma, len(flattened))
        self.data = flattened_perturbed.reshape(self.data.shape)

    def thinning(self, survival_rate=None, number_removal=None, save_path=None, style="homcloud", inplace=False, is_removable=None):
        """
        Thins the point cloud according to the specified survival rate and thinning style.

        Parameters
        ----------
        survival_rate : float
            The proportion of points to retain in the point cloud.
        number_removal : int, optional
            Number of removal iterations to perform. Defaults to 10.
        inplace : bool, optional
            If True, thinning is performed in place. Defaults to True.
        save_path : str, optional
            File path to save the thinned data. If None, data is not saved.
        is_removable : function or None, optional
            Function to determine if a point can be removed. If None, all points are removable.
        style : str, optional
            Thinning style, can be 'homcloud' or 'survived'. Defaults to 'homcloud'.

        Returns
        -------
        Various
            Depending on the thinning style, returns a file path, thinned data, or a new Points3D instance.
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