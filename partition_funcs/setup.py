from distutils.core import setup
name='partition_funcs'

setup(name=name,
      version='0.1',
      py_modules=[name],       # modules to be installed
      scripts=[name + '.py'],  # programs to be installed
      )
