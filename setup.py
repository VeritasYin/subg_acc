from distutils.core import setup, Extension
import numpy

module1 = Extension('gcom_acc',
                    sources = ['gcom_acc.c'],
                    extra_compile_args=['-fopenmp'],
                    extra_link_args=['-lgomp'],
                    include_dirs=[numpy.get_include()])

setup (name = 'GComAcc',
       version = '1.0',
       description = 'This is a package for accelerated graph operations.',
       ext_modules = [module1])
