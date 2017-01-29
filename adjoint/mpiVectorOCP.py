from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time1

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector
from parallelOCP import interval_partition

class MpiVectorOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,
                 Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J

        self.comm = MPI.COMM_WORLD

    def parallel_ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        
        comm = self.comm
        
        rank = comm.Get_rank()
        
        T = self.T        
        dt = float(T)/N
        y = interval_partition(N+1,m,rank)

        if rank == 0:
            y[0] = self.y0            
        else:
            y[0] = u[-1]        #### OBS!!!! ####
            

        for j in range(len(y)-1):            
            y[j+1] = self.ODE_update(y,u,j,j,dt)

        return y


        
            

