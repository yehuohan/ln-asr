#!/usr/bin/env python3

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import glob
import numpy

setup(
    ext_modules=cythonize([Extension(
        'pywvad',
        sources=['pywvad.pyx'] + glob.glob('libfvad/*/*.c'),
        include_dirs=['libfvad', numpy.get_include()])])
)
