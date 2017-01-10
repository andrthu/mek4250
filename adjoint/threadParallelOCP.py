from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from parallelOCP import partition_func,v_comm_numbers,u_part,interval_partition
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time

from optimalContolProblem import OptimalControlProblem, Problem1
from pathos.multiprocessing import ProcessingPool


class TPOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,parallel_J=None,
                Lbfgs_options=None,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.parallel_J=parallel_J



    def thread_parallel_ODE_solver(self,u,N,m):


        variables = []

        for i in range(m):

            variables.append((i,u,N,m))

        p = ProcessingPool(m)
        y = p.map(self.p_ode_solver,variables)

        return y

    def p_ode_solver(self,variables):

        i = variables[0]
        u = variables[1]
        N = variables[2]
        m = variables[3]
        
        dt = float(self.T)/N

        y = interval_partition(n,m,i)

        if i == 0:
            y[0] = self.y0
        else:
            y[0] = u[N+i]

        start = u_part(N+1,m,rank)
                
        for j in range(len(y)-1):
            
            y[j+1] = self.ODE_update(y,u,j,start+j,dt)

        return y
    
    def initial_penalty(self,y,u,mu,N,i):
        
        return mu*(y[i][-1]-u[N+i+1])


    def thread_parallel_adjoint_solver(self,u,N,m,mu):

        y = self.thread_parallel_ODE_solver(u,N,m)

        variables = []

        for i in range(m):

            variables.append((i,y,N,m,mu))


        p = ProcessingPool(m)

        l = p.map(self.p_adjoint_solver,variables)

        return l


    def p_adjoint_solver(self,variables):

        i  = variables[0]
        y  = variables[1]
        N  = variables[2]
        m  = variables[3]
        mu = variables[4]
        dt = float(self.T)/N

        l = interval_partition(n,m,i)

        if i == m-1:
            l[-1] = self.initial_adjoint(y[-1][-1])
        else:
            l[-1] = self.initial_penalty(y,u,mu,N,i)

        for j in range(len(l)-1):
            l[-(j+2)] = self.adjoint_update(l,y[i],j,dt)

        return l
