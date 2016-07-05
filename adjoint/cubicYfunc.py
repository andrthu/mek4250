from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz



class CubicY(Problem1):

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):
        Problem1.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power

        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**p

    
if __name__ == '__main__':

    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    P  = 3
    
    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**power)

    def grad_J(u,p,dt):
        return dt*(u+p)
    
    Problem = CubicY(y0,yT,T,a,P,J,grad_J)
