import numpy as np
import pandas as pd
from ODE_pararealOCP import SimplePpcProblem,PararealOCP


def tes1():

    y0 = 1
    yT = 30
    T  = 1
    a  = 4
    N = 1000
    m = 100

    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        """
        grad =np.zeros(len(u))
        grad[0] = 0.5*u[0]
        grad[1:-1] = u[1:-1]+p[1:-1]
        grad[-1] = 0.5*u[-1]+p[-1]
        return dt*grad
        #"""
        return dt*(u+p)


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    problem.PPCLBFGSsolve(N,m,[1])
tes1()
