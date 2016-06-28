from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz



class Explicit_quadratic(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        #return (y[i] +dt*u[j+1])/(1.-dt*a)

        return y[i-1] + dt*(a*y[i-1]**2 + u[j])


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]

        return (1+2*dt*y[-(i+1)])*l[-(i+1)]



if __name__ == "__main__":

    from matplotlib.pyplot import *

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N1 = 10000
    N2 = 5000
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)


    res1=problem.solve(N1)
    res2=problem.solve(N2)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    
    plot(t1,res1['control'])
    plot(t2,res2['control'],"r--")
    show()
