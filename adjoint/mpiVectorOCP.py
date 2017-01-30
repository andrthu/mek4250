from my_bfgs.lbfgs import Lbfgs
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

class MpiVectorOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,parallel_grad_J=None,
                 Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J
        self.parallel_grad_J = parallel_grad_J
        self.comm = MPI.COMM_WORLD

    def parallel_ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        
        comm = self.comm
        
        rank = comm.Get_rank()
        
        T = self.T        
        dt = float(T)/N
        y = interval_partition(N+1,m,rank)

        if rank == 0:
            y[0] = self.y0     
            j_help = 0
        else:
            y[0] = u[-1]        #### OBS!!!! ####
            j_help = -1

        for j in range(len(y)-1):            
            y[j+1] = self.ODE_update(y,u,j,j+j_help,dt)

        return y

    def parallel_penalty_functional(self,u,N,mu):

        comm = self.comm

        m = comm.Get_size()
        rank = comm.Get_rank()
        y = self.parallel_ODE_penalty_solver(u,N,m)

        return self.parallel_J(u,y,self.yT,self.T,N,mu,comm)

    def initial_penalty(self,y,u,mu,N,rank):

        return mu*(y[-1]-u)
        
    def parallel_adjoint_penalty_solver(self,u,N,m,mu):
        """
        Solving the adjoint equation using finite difference, and partitioning
        the time interval.

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """
        comm = self.comm
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        rank = comm.Get_rank()
        
        l = interval_partition(N+1,m,rank)#partition_func(N+1,m)
        y= self.parallel_ODE_penalty_solver(u,N,m)
        
        if rank == m-1:
            comm.send(y[0],dest=rank-1)
            lam = None
        elif rank!=0:
            lam = comm.recv(source=rank+1)
            comm.send(y[0],dest=rank-1)
        else:
            lam = comm.recv(source=rank+1)

        if rank == m-1:
            l[-1] = self.initial_adjoint(y[-1])
        else:
            l[-1]=self.initial_penalty(y,lam,mu,N,rank) #my*(y[i][-1]-u[N+1+i])
            
        
        for j in range(len(l)-1):
            l[-(j+2)] = self.adjoint_update(l,y,j,dt)
        
        return l

    def penalty_grad(self,u,N,m,mu):

        comm = self.comm
        rank = comm.Get_rank()
        p = self.parallel_adjoint_penalty_solver(u,N,m,mu)           

            
            
        grad = self.parallel_grad_J(u,p,float(self.T)/N,comm)
            
                       
        if rank == 0:
            my_lam = np.array([0])
            comm.send(p[-1],dest=1)
        elif rank!= m-1:
            p_n = comm.recv(source=rank-1)
            comm.send(p[-1],dest=rank+1)

            my_lam = np.array([p[0]-p_n])
            grad[-1] = p[0]-p_n
        else:
            p_n = comm.recv(source=rank-1)
            my_lam = np.array([p[0]-p_n])
            grad[-1] = p[0]-p_n
            
        return grad



class simpleMpiVectorOCP(MpiVectorOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,parallel_J=None,parallel_grad_J=None,options=None):

        MpiVectorOCP.__init__(self,y0,yT,T,J,grad_J,options=options,
                              parallel_J=parallel_J,
                              parallel_grad_J=parallel_grad_J)

        self.a = a

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a)


def generate_problem(y0,yT,T,a):
    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0.5*dt*(u[0]) 
        grad[-1] = 0.5*dt*(u[-1]) + dt*p[-2]
        return grad
        return dt*(u+p)

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
            s[0] = 0.5*((y[-1]-yT)**2) 
            
            I = dt*np.sum(u[:-2]**2)
            I+= dt*0.5*u[-2]**2
            s[0] += 0.5*I 
        elif rank!=0:
            s[0] = 0.5*dt*np.sum(u[:-1]**2)
            

            s[0] += 0.5*mu*(y[-1]-lam)**2

        else:
            s[0] = 0.5*dt*np.sum(u[1:]**2)
            
            s[0] += 0.5*0.5*dt*u[0]**2
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
            grad[1:] = dt*(u[1:] + p[:-1])
            grad[0] = 0.5*dt*u[0]

        elif rank!=m-1:
            grad = np.zeros(len(u))

            grad[:-1]=dt*(u[:-1]+p[:-1])
        else:

            grad = np.zeros(len(u))

            grad[:-2] = dt*(u[:-2]+p[:-2])
            grad[-2] = dt*0.5*u[-2] + dt*p[-2]
        return grad
        

        

    
    
    pproblem = simpleMpiVectorOCP(y0,yT,T,a,J,grad_J,parallel_J=parallel_J,parallel_grad_J=parallel_grad_J)
    problem = Problem1(y0,yT,T,a,J,grad_J)

    return problem,pproblem


def local_u_size(n,m,i):

    q = n/m
    r = n%m

    if i == 0:
        if r>0:
            return q+1
        else:
            q
    else:
        if r-i>0:
            return q+2
        else:
            return q+1

    

def test_mpi_solvers():
    y0 = 1
    yT = 1
    T =  1
    a =  1

    non_mpi, mpi_problem = generate_problem(y0,yT,T,a)
    
    comm = mpi_problem.comm
    

    N = 100
    m = comm.Get_size()
    rank = comm.Get_rank()

    u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
    #print local_u_size(N+1,m,rank),rank

    print
    if rank==1:
        u[-1]=0
    u2 = np.zeros(N+m)
    u2[-1]=0
    y = mpi_problem.parallel_ODE_penalty_solver(u,N,m)
    y2,Y = non_mpi.ODE_penalty_solver(u2,N,m)

    if rank ==0:
        print y-y2[0],rank
    if rank==1:
        print y -y2[1],rank
    mu =1
    print
    #comm.Barrier()
    p = mpi_problem.parallel_adjoint_penalty_solver(u,N,m,1)
    p2, P = non_mpi.adjoint_penalty_solver(u2,N,m,1)
    if rank ==0:
        print p-p2[0],rank
    if rank ==1:
        print p-p2[1],rank
    print
    print mpi_problem.parallel_penalty_functional(u,N,1),non_mpi.Penalty_Functional(u2,N,m,1)
    
    grad = mpi_problem.penalty_grad(u,N,m,mu)
    print

    Js,grad_Js=non_mpi.generate_reduced_penalty(float(T)/N,N,m,mu)

    grads = grad_Js(u2)
    if m==2:
        if rank == 0:
            print grad-grads[:len(grad)],rank,len(grad)
        if rank == 1:
            print grad-grads[len(grad):],rank,len(grad)


    

if __name__ =='__main__':
    test_mpi_solvers()
    
