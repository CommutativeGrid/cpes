import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.5.0"
PACKAGE_NAME = "cpes"
AUTHOR = "初春飾利"
AUTHOR_EMAIL = "xucgphi@gmail.com"
URL = ""

LICENSE = "MIT"
DESCRIPTION = "Close-packing of equal spheres"
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")
LONG_DESC_TYPE = "text/markdown"

CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.10",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = [
    "numpy>=1.21.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.2.2",
    "rdfpy>=1.0.0",
    "matplotlib>=3.8.0",
    "requests>=2.28.0",
    "plotly>=5.15.0",
    "pyvista>=0.39.0",
    "pythreejs>=2.4.0",
    "panel>=0.14.0",
    "crystals>=1.6.0",
    "hexalattice==1.3.0"
]

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    classifiers=CLASSIFIERS,
    author=AUTHOR,
    license=LICENSE,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    package_data={"cpes": ["isaacs_data/*"]},
)
