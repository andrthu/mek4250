from mpi4py import MPI

comm = MPI.COMM_WORLD

rank = comm.Get_rank()
size = comm.Get_size()
if rank ==0:
    test = 10*size
else:
    test = 10
print "hello from p =",rank
MPI.Finalize()
print "hello from test =",test
