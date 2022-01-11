import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = "0.1.0"
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
    "Programming Language :: Python :: 3.9",
    "Topic :: Scientific/Engineering",
]

INSTALL_REQUIRES = [
    "numpy",
    "sklearn",
    "rdfpy",
    "matplotlib",
    "hexalattice",
    "requests",
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
)
