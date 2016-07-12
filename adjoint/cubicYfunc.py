from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz



class GeneralPowerY(Problem1):
    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/p*(y(T)-yT)**p
    with y' = ay + u
    """

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):
        Problem1.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power
    
        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**(p-1)
    
class CubicY(Problem1):
    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/3*(y(T)-yT)**3
    with y' = ay + u
    """
    def __init__(self,y0,yT,T,a,J,grad_J,options=None):
        Problem1.__init__(self,y0,yT,T,a,J,grad_J,options)
    
    def initial_adjoint(self,y):
        
        p = 3
        return (y - self.yT)**(p-1)
    
if __name__ == '__main__':

    from matplotlib.pyplot import *

    y0 = 1
    yT = 0
    T  = 1
    a  = 1
    P  = 3
    N=700
    
    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return (0.5*I + (1./power)*(y-yT)**power)

    def J2(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)
    
    problem  = GeneralPowerY(y0,yT,T,a,P,J,grad_J)
    problem2 = Problem1(y0,yT,T,a,J2,grad_J)

    res1 = problem.plot_solve(N,state=True)
    print 
    res2 = problem2.plot_solve(N,state=True)

    t = np.linspace(0,T,N+1)


    plot(t,res1['control'].array())
    plot(t,res2['control'].array(),'r--')
    show()

    problem.simple_test(N)
    
