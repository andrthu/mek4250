from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()

s = np.zeros(1)+rank+1
S = np.zeros(1)

comm.Reduce(s,S,op=MPI.SUM,root=0)

print S,rank
