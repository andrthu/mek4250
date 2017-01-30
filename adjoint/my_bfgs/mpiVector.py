import numpy as np
from mpi4py import MPI

class MPIVector():

    def __init__(self,vec,comm):
        self.local_vec = vec
        self.comm = comm

    def __add__(self,other):
        return MPIVector(self.local_vec + other.local_vec,self.comm)

    def __sub__(self,other):
        return MPIVector(self.local_vec - other.local_vec,self.comm)

    def __mul__(self,a):
        
        return MPIVector(a*self.local_vec,self.comm)
    __rmul__=__mul__

    def dot(self,other):
        comm = self.comm
        local_res = np.array([np.sum(self.local_vec[:]*other.local_vec[:])])
        print local_res,comm.Get_rank()
        global_res = np.zeros(1)
        comm.Allreduce(local_res,global_res,op=MPI.SUM)
        return global_res[0]
    def __str__(self):
        return str(self.local_vec)
    def __len__(self):
        return len(self.local_vec)
    def __getitem__(self,i):
        return self.local_vec[i]
    def __setitem__(self,i,val):
        self.local_vec[i] = val
def test_mpivec():

    comm = MPI.COMM_WORLD
    n = 10
    rank = comm.Get_rank()

    a = (rank+1)*(np.zeros(n)+1)
    b = rank*(np.zeros(n)+1)
    v = MPIVector(a,comm)
    u = MPIVector(b,comm)
    
    if rank==0:
        print 'sum'
    print u+v
    comm.Barrier()
    print
    if rank==0:
        print 'scalar multiplication:'
    print 2*v
    comm.Barrier()
    print
    if rank==0:
        print 'dot prod:'
    print v.dot(v)
    
    comm.Barrier()
    print
    if rank==0:
        print 'length:'
    print len(v)


if __name__=='__main__':
    test_mpivec()
    
