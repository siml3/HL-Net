from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name="cython_nms", ext_modules=cythonize('cython_nms.pyx'), include_dirs=[numpy.get_include()])