#!/usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension(
        'pyvad',
        sources=['pyvad.pyx', 'src/wb_vad.c'], 
        include_dirs=['src', numpy.get_include()])])
)
