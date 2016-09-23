%module sample

%{
#define SWIG_FILE_WITH_INIT
#include "sample.h"
%}



%include "numpy.i" 

%init %{
  import_array();
%}

%apply (int* ARGOUT_ARRAY1, int DIM1)  {(int* x,int Nx)}
%include "sample.h"
