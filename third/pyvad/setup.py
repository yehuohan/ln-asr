#!/usr/bin/env python
# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize([Extension(
        'pyvad',
        sources=['pyvad.pyx', 'wb_vad.c'],
        include_dirs=['.', numpy.get_include()])])
)
