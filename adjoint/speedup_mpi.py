from mpiVectorOCP import simpleMpiVectorOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector
from parallelOCP import interval_partition,v_comm_numbers
from ODE_pararealOCP import PararealOCP

class GeneralPowerEndTermMPIOCP(simpleMpiVectorOCP):

    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/p*(y(T)-yT)**p
    with y' = ay + u
    """

    def __init__(self,y0,yT,T,a,power,J,grad_J,parallel_J=None,parallel_grad_J=None,options=None):
        simpleMpiVectorOCP.__init__(self,y0,yT,T,a,J,grad_J,parallel_J=parallel_J,
                                    parallel_grad_J=parallel_grad_J,options=options)
        self.power = power
    
        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**(p-1)

def interval_length_and_start(N,m,i):

    q = N/m
    r = N%m
    
    start = 0
    end = 0
    j = 0
    while j <=i:
        start = end

        if r-j>0:
            end +=q+1
        else:
            end += q
        j+=1

    return start,end

def non_lin_problem(y0,yT,T,a,p,c=0,func=None):
    
    if func==None:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-c)**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-c+p[:-1])
            grad[0] = 0.5*dt*(u[0]-c)
            grad[-1] = 0.5*dt*(u[-1]-c) + dt*p[-2]
            return grad



        def parallel_J(u,y,yT,T,N,mu,comm):
            m = comm.Get_size()
            rank = comm.Get_rank()
        
        
            dt = float(T)/N
        

            if rank == m-1:
                comm.send(u[-1],dest=rank-1)
                lam = None
            elif rank!=0:
                lam = comm.recv(source=rank+1)
                comm.send(u[-1],dest=rank-1)
            else:
                lam = comm.recv(source=rank+1)

        
            s = np.zeros(1)
            if rank == m-1:
                s[0] = ((y[-1]-yT)**p)/ float(p)
            
                I = dt*np.sum((u[:-2]-c)**2)
                I+= dt*0.5*(u[-2]-c)**2
                s[0] += 0.5*I 
            elif rank!=0:
                s[0] = 0.5*dt*np.sum((u[:-1]-c)**2)
                s[0] += 0.5*mu*(y[-1]-lam)**2

            else:
                s[0] = 0.5*dt*np.sum((u[1:]-c)**2)            
                s[0] += 0.5*0.5*dt*(u[0]-c)**2
                s[0] += 0.5*mu*(y[-1]-lam)**2
        
            S =np.zeros(1)
            comm.Barrier()
            comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

            return S[0]


        def parallel_grad_J(u,p,dt,comm):

            rank = comm.Get_rank()
            m = comm.Get_size()
            if rank ==0:

                grad = np.zeros(len(u))
                grad[1:] = dt*(u[1:] + p[:-1]-c)
                grad[0] = 0.5*dt*(u[0]-c)

            elif rank!=m-1:
                grad = np.zeros(len(u))

                grad[:-1]=dt*(u[:-1]+p[:-1]-c)
            else:

                grad = np.zeros(len(u))
                grad[:-2] = dt*(u[:-2]+p[:-2]-c)
                grad[-2] = dt*0.5*(u[-2]-c) + dt*p[-2]
            return grad

    else:
        def J(u,y,yT,T,power):
            t = np.linspace(0,T,len(u))

            I = trapz((u-func(t))**2,t)

            return 0.5*I + (1./power)*(y-yT)**power

        def grad_J(u,p,dt):
            t = np.linspace(0,T,len(u))
            grad = np.zeros(len(u))
            grad[1:] = dt*(u[1:]-func(t[1:])+p[:-1])
            grad[0] = 0.5*dt*(u[0]-func(t[0]))
            grad[-1] = 0.5*dt*(u[-1]-func(t[-1])) + dt*p[-2]
            return grad


        def parallel_J(u,y,yT,T,N,mu,comm):
            m = comm.Get_size()
            rank = comm.Get_rank()
        
        
            dt = float(T)/N
            
            t_start,t_end = interval_length_and_start(N+1,m,rank)

            t = dt*np.linspace(t_start,t_end-1,t_end-t_start)

            if rank == m-1:
                comm.send(u[-1],dest=rank-1)
                lam = None
            elif rank!=0:
                lam = comm.recv(source=rank+1)
                comm.send(u[-1],dest=rank-1)
            else:
                lam = comm.recv(source=rank+1)

        
            s = np.zeros(1)
            if rank == m-1:
                s[0] = ((y[-1]-yT)**p)/ float(p)
            
                I = dt*np.sum((u[:-2]-func(t[:-1]))**2)
                I+= dt*0.5*(u[-2]-func(t[-1]))**2
                s[0] += 0.5*I 
            elif rank!=0:
                s[0] = 0.5*dt*np.sum((u[:-1]-func(t[:]))**2)
                s[0] += 0.5*mu*(y[-1]-lam)**2

            else:
                s[0] = 0.5*dt*np.sum((u[1:]-func(t[1:]))**2)            
                s[0] += 0.5*0.5*dt*(u[0]-func(t[0]))**2
                s[0] += 0.5*mu*(y[-1]-lam)**2
        
            S =np.zeros(1)
            comm.Barrier()
            comm.Allreduce(s,S,op=MPI.SUM)#,root=0)

            return S[0]


        def parallel_grad_J(u,p,dt,comm):

            rank = comm.Get_rank()
            m = comm.Get_size()
            N = u.global_length()
            t_start,t_end = interval_length_and_start(N+1-m,m,rank)

            t = dt*np.linspace(t_start,t_end-1,t_end-t_start)
            if rank ==0:

                grad = np.zeros(len(u))
                grad[1:] = dt*(u[1:] + p[:-1]-func(t[1:]))
                grad[0] = 0.5*dt*(u[0]-func(t[0]))

            elif rank!=m-1:
                grad = np.zeros(len(u))

                grad[:-1]=dt*(u[:-1]+p[:-1]-func(t))
            else:

                grad = np.zeros(len(u))
                grad[:-2] = dt*(u[:-2]+p[:-2]-func(t[:-1]))
                grad[-2] = dt*0.5*(u[-2]-func(t[-1])) + dt*p[-2]
            return grad



    problem = GeneralPowerEndTermMPIOCP(y0,yT,T,a,p,J,grad_J,parallel_J=parallel_J,parallel_grad_J=parallel_grad_J)

    return problem


def get_speedup():
    
    import matplotlib.pyplot as plt

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4
    c = 0.5

    
    problem = non_lin_problem(y0,yT,T,a,p,c=c,func=lambda x:x**2)
    comm = problem.comm
    N = 1000
    m = comm.Get_size()
    rank = comm.Get_rank()
    
    t0 = time.time()
    seq_res=problem.solve(N,Lbfgs_options={'jtol':0,'maxiter':10})
    t1 = time.time()
    comm.Barrier()
    t2 = time.time()
    par_res=problem.parallel_penalty_solve(N,m,[N],Lbfgs_options={'jtol':0,'maxiter':10,'ignore xtol':False})
    t3 = time.time()
    
    print 
    print t1-t0,t3-t2,(t1-t0)/(t3-t2), seq_res.niter,par_res[-1].niter,seq_res.lsiter,par_res[-1].lsiter

    x = par_res[-1].x.gather_control()
    if rank == 0:
        plt.plot(x[:N+1])
        plt.plot(seq_res.x,'r--')
        plt.show()

if __name__ == '__main__':
    
    get_speedup()
