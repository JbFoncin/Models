# -*- coding: utf-8 -*-
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
ext_module = Extension(
    "MNPP",
    ["MNPP.pyx", "/home/jb/data_analysis/GHK_tools/Truncated_normal_generator.cpp" ],
    extra_compile_args=['-fopenmp', '-std=c++11', '-O3'],
    extra_link_args=['-fopenmp'],
    language="c++",
    include_dirs=["/home/jb/data_analysis/eigen-eigen-dc6cfdf9bcec/Eigen/",
                  "/home/jb/data_analysis/boost/boost_1_61_0/",
                  "/home/jb/data_analysis/GHK_tools/"]
    )

setup(
    name = 'Hello world app',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module])