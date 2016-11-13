from ODE_pararealOCP import PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem
import time

class Problem3(PararealOCP):
    """
    optimal control with ODE y=ay'+u
    and J(u)=||u||**2 + alpha*(y(T)-yT)**2
    """

    def __init__(self,y0,yT,T,a,alpha,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a
        self.alpha = alpha
        
        def JJ(u,y,yT,T):
            return J(u,y,yT,T,self.alpha)
        def grad_JJ(u,p,dt):
            return grad_J(u,p,dt,self.alpha)
        self.J=JJ
        self.grad_J = grad_JJ


    def initial_adjoint(self,y):
        return self.alpha*(y - self.yT)

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a)

def test1():

    y0 = 1.2
    a = 0.9
    T = 1.
    yT = 5
    alpha = 0.5
    N = 1000
    m = 10
    
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)

    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)

    problem.PPCSDsolve(N,m,[1,10,50,100])[-1]


test1()

