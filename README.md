# Close-packing of equal spheres
# 球の正規最密充填

This package provides the following three main functions:

* Construct fcc/hcp lattice layer by layer
* Construct fcc/hcp lattice via building conventional cells.
* Load the fcc/hcp coordinates of atoms from [ISAACS](http://isaacs.sourceforge.net/ex.html#Si)

## Installation

```
git clone https://github.com/Commutative-Ladders/cpes.git
```
then `cd` to the cloned foler, and
```
pip install .
```

To upgrade, run
```
pip install . --upgrade
```

### Installation in developement (editable) mode

> Deploy your project in “development mode”, such that it’s available on sys.path, yet can still be edited directly from its source checkout.

```
pip install -e .
```

## Basic usage

### Generate a fcc/hcp lattice layer by layer

Build a 10x10x10 hcp lattice, with sphere radius 1.0. Generated data is nearly centered, has an atom at (0,0,0), with the normal vector of any added layer points upwards vertically.

```python
from cpes import face_centered_cubic, hexagonal_close_packing
import numpy as np

fcc_generated=face_centered_cubic(10,radius=1.0) 

print(fcc_generated.data) # access the coordinates
np.savetxt('fcc.xyz',fcc.data) # save the coordinates to a file
coords=np.loadtxt('fcc.xyz') # load coordinates from a file
```
Execute a thinning process with survival rate 0.5, and save the file in accordan with the format of homcloud.
```python
fcc_generated.thinning(survival_rate=0.5,save=True,style='homcloud')
```



### Load data from [ISAACS](http://isaacs.sourceforge.net/ex.html)

Load the Cartesian coordinates of Au.
```python
from cpes import fcc_au_cart

fcc_au=fcc_au_cart(mode='online')
fcc_au.original # access the original cartesian coordinates
fcc_au.data # access the normalized coordinates (such that the sphere radius becomes 1.0)
```

Check that lists of distance of the k-th nearest neighbor are the same for the generated data and the loaded data.

```python
[i-j<1e-3 for i,j in zip(fcc_au.distance_array(),fcc_generated.distance_array())]
```

## Methods

`.plot`: plot the structure.
`.rdf_plot`: plot the radial distribution function. Notice that the implementation of this function is currently translation and orientation sensitive.
`.thinning`: compute the thinning of `self.data` with the provided `survival_rate`.
`.sphere_confine`: return the atoms within the given radius of the origin.

