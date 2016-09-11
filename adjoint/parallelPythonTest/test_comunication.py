from mpi4py import MPI
import numpy as np


comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:

    init = [10*(i+1) for i in range(size)]
    
    result = np.empty(10*size,dtype='i')
else:
    init=None
    result=None
init = comm.scatter(init,root=0)
print init,rank

new_data = np.empty(10,dtype='i')

for i in range(len(new_data)):
    new_data[i] = init + i

comm.Gather(new_data,result,root=0)
if rank==0:
    print result

