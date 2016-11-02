from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem


class PararealOCP(OptimalControlProblem):


    def __init__(self,y0,yT,T,J,grad_J,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options=options)
        


    def adjoint_propogator(self,m,delta0,S):

        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[-1] = delta0

        for i in range(m):

            delta[-(i+2)]=self.adjoint_propogator_update(delta,S,i,dT)

        return delta

    def ODE_propogator(self,m,delta0,S):
        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[0] = delta0

        for i in range(m):
            delta[i+1] = self.ODE_update(delta,S/dT,i,i-1,dT)

        return delta

    def PC_maker(self,N,m):


        def pc(x):
            S = np.zeros(m)
            S[:-1] = x[N+1:].copy()

            delta =self.adjoint_propogator(m,0,S)

            for i in range(len(S)):
                S[i] = S[i]+delta[i+1]

            S2 = np.zeros(m)
            S2[1:]=S[:-1].copy()

            delta2 = self.ODE_propogator(m,delta[0],S2)

            for i in range(len(S)):
                S2[i] = S2[i]+delta2[i]

            x[N+1:]=S2[1:]
            
            return x

        return pc

    def PPCSDsolve(self,N,m,my_list,x0=None,options=None):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        
        result = []
        PPC = self.PC_maker(N,m)
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_SD_options(options)
            SDopt = self.SD_options

            Solver = PPCSteepestDecent(J,grad_J,x0.copy(),PPC,
                                       options=SDopt)
            res = Solver.solve()

            result.append(res)
        if len(result)==1:
            return res
        else:
            return result


    def adjoint_propogator_update(self,l,rhs,i,dt):
        raise NotImplementedError,'not implemented'

class SimplePpcProblem(PararealOCP):
    """
    optimal control with ODE y'=ay+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a) 


    def adjoint_propogator_update(self,l,rhs,i,dt):
        a = self.a
        return (l[-(i+1)]+rhs[-(i+1)])/(1-dt*a)


if __name__ == "__main__":
    from matplotlib.pyplot import *

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


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    
    N = 1000
    m = 10

    #res = problem.PPCSDsolve(N,m,[1])

    res2 = problem.scipy_solver(N,disp=True)
    
    res3 = problem.solve(N,algorithm='my_steepest_decent')
    





"""
-(l[-(i+1)]-l[(i+2)])/dT = a*l[-(i+2)] + S[-(i+1)]/dT

l[-(i+2)](1-a*dT) = l[-(i+1)] + S[-(i+1)]

l[-(i+2)] = (l[-(i+1)]+S[-(i+1)])/(1-dT*a) 
"""
