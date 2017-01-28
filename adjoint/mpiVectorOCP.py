from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector


class MpiVectorOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,
                 Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J

        self.comm = MPI.COMM_WORLD
