import numpy as np

def partition_func(n,m):
    """
    returns a list of arrays, that are a m-decomposition of a
    lenght n array
    """
    N=n/m
    rest = n%m
    partition = []

    if rest>0:
        partition.append(np.zeros(N+1))
    else:
        partition.append(np.zeros(N))

    for i in range(1,m):
        if rest-i>0:
            partition.append(np.zeros(N+2))
        else:
            partition.append(np.zeros(N+1))

    return partition

def v_comm_numbers(n,m):

    N = n/m
    r = n%m

    scatter_s = [0]
    scatter_l = []
    gather_s = [0]
    gather_l = []

    if r>0:
        scatter_l.append(N + 1)
        gather_l.append(N + 1)
    else:
        scatter_l.append(N)
        gather_l.append(N)
    
    for  i in range(1,m):
        if r -i >0:
            scatter_l.append(N+2)
            gather_l.append(N+1)
        else:
            scatter_l.append(N+1)
            gather_l.append(N)

        scatter_s.append(scatter_s[i-1] + scatter_l[i-1]-1)
        gather_s.append(scatter_s[i]+1)
        
    return tuple(scatter_s),tuple(scatter_l),tuple(gather_s),tuple(gather_l)

def u_part(n,m,i):

    
    N = n/m
    rest = n%m
    if i==0:
        return 0

    if rest>0:
        start = N
    else:
        start = N-1
    for j in range(i-1):
        if rest-(j+1)>0:
            start += N+1
        else:
            start += N

    return start

def interval_partition(n,m,i):
    
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = np.zeros(N+1)
        else:
            state = np.zeros(N)
    else:
        if rest - i >0:
            state = np.zeros(N+2)
        else:
            state = np.zeros(N+1)
    return state


def int_par_len(n,m,i):
    """
    With fine resolution n and m time decomposition intervals,
    functions find number of points in interval i
    """
    N=n/m
    rest=n%m

    if i==0:
        if rest>0:
            state = N+1
        else:
            state = N
    else:
        if rest - i >0:
            state = N+2
        else:
            state = N+1
    return state

def gather_y(y,N):
    """
    Gathers tje y arrays into one array Y
    """
    Y = np.zeros(N+1)
    
    start = len(y[0])
    Y[:start] = y[0][:]
    
    for i in range(len(y)-1):
        
        end = start + len(y[i+1])-1
        
        Y[start:end] = y[i+1][1:]
        start = end
    return Y

def time_partition(start,end,Tn,m):

    N = np.zeros(m)
    T = np.zeros(m+1) 

    timestep = float(end-start)/Tn

    fraq = Tn/m
    rest = Tn%m

    t = start
    T[0] = t

    for i in range(0,m):
        if rest-i >0:
            N[i] = fraq +1
        else:
            N[i] = fraq 

        t = t + timestep*N[i]
        T[i+1] = t

    return N,T
def partition_start(Tn,m):
    
    
    fraq = Tn/m
    rest = Tn%m
    
    start = []
    start.append(0)
    
    for i in range(m-1):
        if rest - i >0:
            start.append(start[i] + fraq + 1)
        else:
            start.append( start[i] + fraq) 
    return start
