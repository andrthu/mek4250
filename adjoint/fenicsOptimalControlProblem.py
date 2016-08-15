from dolfin import *
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np


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
            N[i] = fraq + 1
        else:
            N[i] = fraq

        t = t + timestep*N[i]
        T[i+1] = t

    return N,T


class FenicsOptimalControlProblem():

    def __init__(self,V,mesh):

        self.V = V
        self.mesh = mesh
        self.T = T
        


    def initial_condition(self,control,opt):
        raise NotImplementedError, 'initial_condition not implemented'

    def adjoint_ic(self,opt):
        raise NotImplementedError, 'adjoint_ic not implemented'

    def PDE_form(self,control,opt,u,u_,v,timestep):
        raise NotImplementedError, 'PDE_form not implemented'

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        raise NotImplementedError, 'adjoint_form not implemented'


    def PDE_solver(self,control,opt,start,end,Tn):

        U = []

        u_ = self.initial_condition(control,opt)

        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)

        timestep = Constant((end-start)/float(Tn))

        F,bc = self.PDE_form(control,opt,u,u_,v,timestep)


        t  = start

        while (t<end - DOLFIN_EPS):
            solve(F==0,u,bc)
            u_.assign(u)
            U.append(u_.copy())

            t += float(timestep)

        return U

    def adjoint_interval_solver(self,opt,p_ic,U,start,end,Tn):
        
        P = []

        timestep = Constant((end-start)/float(Tn))

        
        p_ = p_ic.copy()
        P.append(p_.copy())

        p = Function(self.V)
        v = TestFunction(self.V)
        u = U[-1]
        
        F,bc = self.adjoint_form(opt,u,p,p_,v,timestep)

        t = end
        count = -2
        while ( t> start + DOLFIN_EPS):

            solve(F==0,p,bc)
            p_.assign(p)
            u.assign(U[count].copy())
        
            P.append(p_.copy())
            count = count -1
            t = t - float(timestep)
    
        return P

        
    def adjoint_solve(self,control,opti,start,end,Tn):

        U = self.PDE_solver(control,opt,start,end,Tn)
        p_ic = self.adjoint_ic(opt)
        
        return self.interval_adjoint(opt,p_ic,U,start,end,V,Tn)


    def penalty_PDE_solver(self,control,opt,lam_ic,start,end,Tn,m):

        N,T = time_partition(start,end,Tn,m)
