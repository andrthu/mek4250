from mpi4py import MPI
import numpy as np

N = 11
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    x = np.linspace(0,100,N)
    x2 = np.zeros(N)
    x3 = np.zeros(N)
else:
    x  = None
    x2 = None
    x3 = None

start=[0]
l = []
l2 = []
r = N%size
n=N/size
if r>0:
    l.append(n + 1)
    l2.append(n + 1)
else:
    l.append(n)
    l2.append(n)

for i in range(1,size):
    if r -i >0:
        l.append(n+2)
        l2.append(n+1)
    else:
        l.append(n+1)
        l2.append(n)

    start.append(start[i-1] + l[i-1]-1)

if rank==0:
    print l, start 

xlocal = np.zeros(l[rank])

comm.Scatterv([x,tuple(l),tuple(start),MPI.DOUBLE],xlocal)

print xlocal,rank, 'lel'



start2 = start
start3 = [i+1 for i in start]
start3[0] = 0
#start3[1:] = [i+1 for i in start[1:]]

l3 = [i for i in l2]
l2[0] = l2[0]-1
l2[-1] = l2[-1]+1
comm.Barrier()
if rank==size-1:
    x_new1 = 10*xlocal.copy()
    
else:
    x_new1 = 10*xlocal[:-1].copy()

if rank==0:
    x_new2 = 10*xlocal.copy()
    
else:
    x_new2 = 10*xlocal[1:].copy()

comm.Gatherv(x_new1,[x2,tuple(l2),tuple(start2),MPI.DOUBLE])
comm.Gatherv(x_new2,[x3,tuple(l3),tuple(start3),MPI.DOUBLE])

if rank==0:
    print x2
    print
    print x3

