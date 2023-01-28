from distutils.core import setup, Extension
import numpy

module1 = Extension('subg_acc',
                    sources = ['subg_acc.c'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp'],
                    include_dirs=[numpy.get_include()])

setup (name = 'SubGAcc',
       version = '2.0',
       description = 'This is an extension library based on C and openmp for accelerating subgraph operations.',
       ext_modules = [module1])
