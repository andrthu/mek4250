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

        self.Lbfgs_options = self.default_Lbfgs_options()
        

    def default_Lbfgs_options(self):
        """
        default options for LGFGS
        """
        default = {"jtol"                   : 1e-4,
                   "maxiter"                : 500,
                   "mem_lim"                : 10,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : True,}

        return default

    def adjoint_ic(self,opt):
        raise NotImplementedError, 'adjoint_ic not implemented'

    def PDE_form(self,ic,opt,u,u_,v,timestep):
        raise NotImplementedError, 'PDE_form not implemented'

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        raise NotImplementedError, 'adjoint_form not implemented'

    def get_control(self,opt,ic,m):
        raise NotImplementedError, 'get_control not implemented'

    def get_opt(self,control,opt,ic,m):
        raise NotImplementedError, 'get_opt not implemented'


    def PDE_solver(self,ic,opt,start,end,Tn):

        U = []

        u_ = ic.copy()

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

        
    def adjoint_solver(self,ic,opti,start,end,Tn):

        U = self.PDE_solver(ic,opt,start,end,Tn)
        p_ic = self.adjoint_ic(opt)
        
        return self.adjoint_interval_solver(opt,p_ic,U,start,end,Tn)


    def penalty_PDE_solver(self,opt,lam_ic,start,end,Tn,m):

        N,T = time_partition(start,end,Tn,m)


        U = []
        V = self.V

        U.append(self.PDE_solver(ic,opt,T[0],T[1],V,N[0]))
        for i in range(1,m):
            U.append(self.PDE_solver(lam_ic[i-1],opt,T[i],T[i+1],N[i]))

        return U

    def penalty_adjoint_solver(self,ic,lam_ic,opti,start,end,Tn,m,mu):

        N,T = time_partition(start,end,Tn,m)
    
        U =  general_burger_solver(ic,lam_ic,start,end,V,Tn,m)
        
        P = []
    
        for i in range(m-1):
            P_ic = project((U[i][-1]-lam_ic[i])*Constant(mu),V)
            sta = T[i]
            en = T[i+1]
            P.append(self.adjoint_interval_solver(opt,P_ic,U[i],sta,en,N[i]))

        P_ic = self.adjoint_ic(opt)
        P.append(interval_adjoint(optP_ic,U[-1],T[-2],T[-1],N[-1]))

        return P

    def solver(self,opt,ic,start,end,Tn,Lbfgs_options=None)):

        def J(x):
            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            U = self.PDE_solver(loc_ic,loc_opt,start,end,Tn)
            return self.J(opt,ic,U,start,end)
        
        def grad_J(x):

            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            P = self.adjoint_solver(loc_ic,loc_opti,start,end,Tn)

            return self.grad_J(P,loc_opt,loc_ic)


        control0 = self.get_control(opt,ic,m)

        if Lbfgs_options==None:
            Loptions = self.Lbfgs_options
        else:
            Loptions = self.Lbfgs_options
            for key, val in Lbfgs_options.iteritems():
                Loptions[key]=val


        solver = Lbfgs(J,grad_J,control0,options=Loptions)

        res = solver.solve()

        return res



