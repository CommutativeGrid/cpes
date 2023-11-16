# Close-packing of equal spheres (球の正規最密充填)

This repository contains the Python package `cpes`, which serves as a dependency for the [commutazzio](https://github.com/CommutativeGrids/commutazzio) project. The `cpes`` package focuses on constructing and manipulating close-packed lattice structures.

## Key Features

* Construction of face-centered cubic (fcc) and hexagonal close-packed (hcp) lattice structures layer by layer.
* * Load the fcc/hcp coordinates of atoms from [ISAACS](http://isaacs.sourceforge.net/ex.html)


## Installation

To install the package, use:
```
pip install .
```
To install the package, use:
```
pip install . --upgrade
```

### Development Mode Installation

For developers who wish to make changes to the package while using it:
```
pip install -e .
```
### Optional Jupyter Extension

If you encounter issues with `pyvista` in Jupyter notebooks, try installing the following extension ([ref](https://github.com/pyvista/pyvista/issues/332)):
```
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```

## Basic usage

### Generate a fcc/hcp lattice layer by layer

Example: Creating a 10x10x10 hcp lattice with a sphere radius of 1.0:

```python
from cpes import FaceCenteredCubic, HexagonalClosePacking
import numpy as np

fcc = FaceCenteredCubic(10, radius=1.0)

print(fcc.data)  # access the coordinates
np.savetxt('fcc.xyz', fcc.data)  # save the coordinates to a file
coords = np.loadtxt('fcc.xyz')  # load coordinates from a file
```

The generated lattice is nearly centered, has an atom at (0,0,0), with the normal vector of any added layer points upwards vertically.

### Data Thinning Process
Example: Applying a thinning process with a 50% survival rate, and save the file in accordance with the format of [homcloud](https://homcloud.dev/index.en.html):
```python
fcc.thinning(survival_rate=0.5, save=True, style='homcloud')
```


### Load data from [ISAACS](http://isaacs.sourceforge.net/ex.html)

Example: Loading Cartesian coordinates of Au:
```python
from cpes import FccAuCart

fcc_au = FccAuCart(mode='online')
print(fcc_au.original)  # access the original cartesian coordinates
print(fcc_au.data)  # access the normalized coordinates (sphere radius normalized to 1.0)
```
Check that distance of the k-th nearest neighbor are the same for the generated data and the loaded data:

```python
[i - j < 1e-3 for i, j in zip(fcc_au.distance_array(), fcc.distance_array())]
```

## Methods

* `.plot`: Visualize lattice structures.
* `.rdf_plot`: Generate radial distribution function plots (susceptible to translation and orientation).
* `.thinning`: compute the thinning of `self.data` with the provided `survival_rate`.
* `.sphere_confine`: Filter atoms within a specified radius.