from mpi4py import MPI
import numpy as np

N = 11
comm = MPI.COMM_WORLD

size = comm.Get_size()
rank = comm.Get_rank()

if rank == 0:
    x = np.linspace(0,100,N)
    x2 = np.zeros(N)
else:
    x = None
    x2 =None

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

print xlocal,rank



start2 = start
start2[1:] = [i+1 for i in start2[1:]]
comm.Barrier()
if rank==0:
    x_new = 10*xlocal.copy()
else:
    x_new = 10*xlocal[1:].copy()
comm.Gatherv(x_new,[x2,tuple(l2),tuple(start2),MPI.DOUBLE])

if rank==0:
    print x2
