from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time
import sys

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector
from parallelOCP import interval_partition,v_comm_numbers
from ODE_pararealOCP import PararealOCP
from mpiVectorOCP import MpiVectorOCP,simpleMpiVectorOCP,generate_problem,local_u_size

def test_func():
    y0 = 1
    yT=1
    T = 1
    a = 1

    problem,pproblem=generate_problem(y0,yT,T,a)

    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    try:
        N = sys.argv[1]
    except:
        N = 1000

    if m == 1:
        u = np.zeros(N+1)+1
        t0 = time.time()
        problem.Functional(u,N)
        t1 = time.time()
    else:
        mu = 10
        u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
        comm.Barrier()
        t0 = time.time()
        pproblem.parallel_penalty_functional(u,N,mu)
        t1 = time.time()
        comm.Barrier()
        loc_time = np.zeros(1)
        loc_time[0] = t1-t0
        if rank == 0:
            time_vec = np.zeros(m)
        else:
            time_vec = None
        loc_size = tuple(np.zeors(m)+1)
        loc_start = tuple(np.linspace(0,m-1,m))
        comm.Gatherv(loc_time,[time_vec,loc_size,loc_start,MPI.DOUBLE])
        
