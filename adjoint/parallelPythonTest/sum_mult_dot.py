from mpi4py import MPI
import numpy as np
import time

def simple_partition(N,m):

    q = N/m
    r = N%m

    length = []
    start = [0]

    for i in range(m):
        
        if r-i>0:
            length.append(q+1)
        else:
            length.append(q)
    for i in range(m-1):
        if r-i>0:
            start.append(start[i]+q+1)
        else:
            start.append(start[i]+q)


    return length,start

    

def vec_sum_avec(u,v,a,comm):

    #comm = MPI.COMM_WORLD

    

    size = comm.Get_size()
    rank = comm.Get_rank()
    N = len(u)
    
    length,start = simple_partition(N,size)
    
    local_res = np.zeros(length[rank])
    
    
    local_res = u[start[rank]:start[rank]+length[rank]] + a*v[start[rank]:start[rank]+length[rank]]

    #print local_res,rank
    
    result = np.arange(N,dtype=np.float64)
    comm.Allgatherv(local_res,[result,tuple(length),tuple(start),MPI.DOUBLE])
    
    return result


def dot_prod(u,v,comm):

    size = comm.Get_size()
    rank = comm.Get_rank()
    N = len(u)
    
    length,start = simple_partition(N,size)

    #loc_range=start[rank]:start[rank]+length[rank]


    loc_s = np.array(u[start[rank]:start[rank]+length[rank]].dot(v[start[rank]:start[rank]+length[rank]]))
    
    S = np.array(0,dtype=np.float64)
    comm.Allreduce(loc_s,S,op = MPI.SUM)

    return S
    

if __name__=='__main__':
    
    N =1000

    u = np.zeros(N) +1

    v = np.zeros(N) +1

    comm = MPI.COMM_WORLD


    t0 = time.time()
    u.dot(v)
    t1=time.time()
    


    t2=time.time()
    dot_prod(u,v,comm)
    t3=time.time()

    print t1-t0,t3-t2,(t1-t0)/(t3-t2)

    #print dot_prod(u,v,comm)

    """
    N =1000
    
    v = np.zeros(N) + 1
    u= np.linspace(0,N-1,N)
    a = 3
    comm = MPI.COMM_WORLD

    t0 = time.time()
    u+a*v
    t1=time.time()
    


    t2=time.time()
    vec_sum_avec(u,v,a,comm)
    t3=time.time()

    print t1-t0,t3-t2,(t1-t0)/(t3-t2)
    """
