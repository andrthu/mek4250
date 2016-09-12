from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem


def partition_func(n,m):

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

class POCP(OptimalControlProblem):


    def parallel_ODE_penalty_solver(self,u,N,m,comm):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        

        
        rank = comm.Get_rank()
        
        T = self.T
        
        dt = float(T)/N
        y = interval_partition(N+1,m,rank)
        print len(y), rank,len(u)

        if rank == 0:
            y[0]=self.y0
            Y = np.zeros(N+1)
            Y_list = []
        else:
            y[0] = u[N+rank]
            Y = None
            Y_list =None

        start=u_part(N+1,m,rank)
                
        for j in range(len(y)-1):
            
            y[j+1] = self.ODE_update(y,u,j,start+j,dt)
        if rank ==0:
            y_send = y
        else:
            y_send = y[1:]
        y=comm.gather(y,root=0)
        #comm.Gather(y_send,Y,root=0)
        if rank==0:
            print y
            print len(y)
        if rank==0:               
            return y

class PProblem1(POCP):
    """
    optimal control with ODE y=ay'+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)] 
if __name__ == "__main__":

    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = PProblem1(y0,yT,T,a,J,grad_J)

    comm = MPI.COMM_WORLD

    m = comm.Get_size()
    rank = comm.Get_rank()
    N = 1000
    u = np.zeros(N+m)
    
    y=problem.parallel_ODE_penalty_solver(u,N,m,comm)
    print y

    t = np.linspace(1,2,100)
    l=[]
    for i in range(3):
        l.append(u_part(100,3,i))
    print trapz(t**2,t)
    print trapz(t[:l[1]+1]**2,t[:l[1]+1]) + trapz(t[l[1]:l[2]+1]**2,t[l[1]:l[2]+1]) +trapz(t[l[2]:]**2,t[l[2]:])
    

