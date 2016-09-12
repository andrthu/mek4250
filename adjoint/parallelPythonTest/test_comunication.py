from mpi4py import MPI
import numpy as np

def func(a,b):
    comm = MPI.COMM_WORLD

    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:

        init = [a*(i+1) for i in range(size)]
    
        result = np.empty(10*size,dtype='i')
    else:
        init=None
        result=None
    init = comm.scatter(init,root=0)
    print init,rank

    new_data = np.empty(10,dtype='i')

    for i in range(len(new_data)):
        new_data[i] = init + b*i

    comm.Gather(new_data,result,root=0)
    if rank==0:
        print result

func(10,1)
