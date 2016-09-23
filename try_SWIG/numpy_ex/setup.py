from distutils.core import setup, Extension
import numpy

n = 'sample'
numpy_include = numpy.get_include()
example_module = Extension('_'+n,
                           sources=[n+'_wrap.c', n+'.c'],
                           include_dirs = [numpy_include],)




setup(name = n,
      version='0.1',
      ext_modules = [example_module],
      py_modules = [n],
)
