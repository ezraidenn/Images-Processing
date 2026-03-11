"""
Setup script for compiling Cython filters.
Run: python setup.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "filters.cython_filters",
        ["filters/cython_filters.pyx"],
        include_dirs=[np.get_include()],
    )
]

setup(
    name="image_processing_filters",
    ext_modules=cythonize(extensions, compiler_directives={
        'boundscheck': False,
        'wraparound': False,
        'cdivision': True,
    }),
    zip_safe=False,
)
