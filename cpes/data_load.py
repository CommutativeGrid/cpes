import os
from tempfile import NamedTemporaryFile

import numpy as np
import inspect, pathlib

from .close_packings_via_layers import ClosePacking
from .points3d import Points3D
from .utils import TempfileFromUrl


#Read the coordinates data downloaded from isaacs http://isaacs.sourceforge.net/ex.html#Si
class ExampleDataIsaacs:
    """
    A class for handling data from ISAACS (Interactive Structure Analysis of Amorphous and Crystalline Systems).

    Parameters
    ----------
    url : str
        The URL from where the coordinate data is downloaded.
    coord_type : str
        The type of coordinates ('cartesian', 'fractional', etc.).
    mode : str
        The mode of operation ('online' or 'offline').
    offline_fp : str, optional
        The file path for offline data loading. Default is None.

    Attributes
    ----------
    original : numpy.ndarray
        The original coordinates loaded from the file.

    Methods
    -------
    __len__()
        Returns the number of coordinate points.

    """
    def __init__(self, url, coord_type, mode, offline_fp=None):
        self.url = url
        self.coord_type = coord_type
        if mode == "online":
            with TempfileFromUrl(url) as fp:
                raw_data = fp.readlines()
            if len(raw_data[-1].split()) <= 1:
                raw_data.pop()  # delete the last line (an emoji or empty)
            with NamedTemporaryFile(delete=False) as fp:
                for _ in raw_data:
                    fp.write(_)
                path = fp.name
        elif mode == "offline":
            dir_path = os.path.dirname(os.path.realpath(__file__))
            data_path = os.path.join(dir_path, "issacs_data")
            print(f"Load data from file stored in {data_path}")
            path = offline_fp

        self.original = np.loadtxt(
            path, skiprows=2, usecols=(1, 2, 3), unpack=False
        )  # store the original coordinates
        if mode == "online":
            os.remove(path)

    def __len__(self):
        return len(self.original)


class FccAuCart(ExampleDataIsaacs, Points3D):
    """
    Represents Face-Centered Cubic (FCC) gold (Au) crystal structure in cartesian coordinates.

    Inherits from ExampleDataIsaacs and Points3D classes.

    Parameters
    ----------
    mode : str, optional
        The mode of operation ('offline' by default).

    Attributes
    ----------
    cartesian : numpy.ndarray
        The cartesian coordinates of the crystal structure.
    normalized : numpy.ndarray
        The normalized coordinates scaled to make the radius of the atom 1.0.
    data : numpy.ndarray
        The data attribute inherited from Points3D, set to normalized coordinates.

    """
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/au-cart.xyz"
        #dir_name should be the place of the package cpes
        # use pathlib package
        dir_name = pathlib.Path(inspect.getfile(self.__class__)).parent
        # use pathlib package
        path = dir_name / "isaacs_data" / "au-cart.xyz"
        # the last line is deleted from the downloaded original file
        ExampleDataIsaacs.__init__(self,url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.cartesian = (
            self.original
        )  # the cartesian coordinates is the same as the original coordinates
        self.normalized = (
            2 * self.cartesian / 2.877925
        )  # rescale to make the radius of the atom 1.0
        # notice that the lattice is not guaranteed to be centerd exactly at the origin
        Points3D.__init__(self, self.normalized)
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in close_packing can be used on self.data


# Need to check the lattice data below
# online calculator for transformation of coordinates
# https://cci.lbl.gov/cctbx/frac_cart.html
class HcpRuFrac(ExampleDataIsaacs, ClosePacking):
    """
    Represents Hexagonal Close-Packed (HCP) ruthenium (Ru) crystal structure in fractional coordinates.

    Inherits from ExampleDataIsaacs and ClosePacking classes.

    Parameters
    ----------
    mode : str, optional
        The mode of operation ('offline' by default).

    Attributes
    ----------
    fractional : numpy.ndarray
        The fractional coordinates of the crystal structure.
    cartesian : numpy.ndarray
        The converted cartesian coordinates from the fractional coordinates.
    normalized : numpy.ndarray
        The normalized coordinates scaled to make the radius of the atom 1.0.
    data : numpy.ndarray
        The data attribute inherited from Points3D, set to normalized coordinates.

    """
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/ru-frac.xyz"
        dir_name = pathlib.Path(inspect.getfile(self.__class__)).parent
        path = dir_name / "isaacs_data" / "ru-frac.xyz"
         # the last line is deleted from the downloaded original file
        ExampleDataIsaacs.__init__(self,url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.fractional = (
            self.original
        )  # the fractional coordinates is the same as the original coordinates
        # Transofrm the fractional coordinates to cartesian coordinates
        # parameters from the website
        a = 43.29
        b = 43.29
        c = 68.5
        # alpha = np.pi / 2
        # beta = np.pi / 2
        # gamma = 2 * np.pi / 3
        # n2 = (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
        # n2=0
        # transformation matrix, see
        # https://chemistry.stackexchange.com/questions/136836/converting-fractional-coordinates-into-cartesian-coordinates-for-crystallography
        # M  = np.array([[a,0,0],
        #                [b*np.cos(gamma),b*np.sin(gamma),0],
        #      [c*np.cos(beta),c*n2,c*np.sqrt(np.sin(beta)**2-n2**2)]])
        # pre-computed some values to reduce errors
        M = np.array([[a, 0, 0], [b * (-0.5), b * np.sqrt(3) / 2, 0], [0, 0, c]])

        self.cartesian = self.fractional @ M
        self.normalized = (
            2 * self.cartesian / 5.29987615
        )  # rescale to make the radius of the atom 1.0
        Points3D.__init__(self, self.normalized)
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in close_packing can be used on self.data


class FccSiFrac(ExampleDataIsaacs, ClosePacking):
    """
    Represents Face-Centered Cubic (FCC) silicon (Si) crystal structure in fractional coordinates.

    Inherits from ExampleDataIsaacs and ClosePacking classes.

    Parameters
    ----------
    mode : str, optional
        The mode of operation ('offline' by default).

    Attributes
    ----------
    fractional : numpy.ndarray
        The fractional coordinates of the crystal structure.
    cartesian : numpy.ndarray
        The converted cartesian coordinates from the fractional coordinates.
    normalized : numpy.ndarray
        The normalized coordinates scaled to make the radius of the atom 1.0.
    data : numpy.ndarray
        The data attribute inherited from Points3D, set to normalized coordinates.

    """
    def __init__(self, mode="offline"):
        url = "http://isaacs.sourceforge.net/tests/si-frac.xyz"
        dir_name = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(
            dir_name, "isaacs_data", "si-frac.xyz"
        )  # the last line is deleted from the downloaded original file
        ExampleDataIsaacs.__init__(self,url, coord_type="cartesian", mode=mode, offline_fp=path)
        self.fractional = (
            self.original
        )  # the fractional coordinates is the same as the original coordinates
        # Transofrm the fractional coordinates to cartesian coordinates
        # parameters from the website
        a = 32.52
        b = 32.52
        c = 32.52
        # alpha = np.pi / 2
        # beta = np.pi / 2
        # gamma = np.pi / 2
        M = np.array([[a, 0, 0], [0, b, 0], [0, 0, c]])
        self.cartesian = self.fractional @ M
        self.normalized = (
            2 * self.cartesian / 4.6939
        )  # rescale to make the radius of the atom 1
        Points3D.__init__(self, self.normalized)
        self.data = self.normalized  # set self.data to be the normalized coordinates
        # now methods in ClosePacking can be used on self.data
