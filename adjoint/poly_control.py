from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz


class PolynomialControl(Problem1):

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):

        
        Problem1.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power
        self.powers = np.linspace(0,power,power+1)

    def polynomial(coef,t):
        
        p = 0
        for i in range(len(coef)):
            p += coef[i]*t**self.powers[i]

        return p       

        
        
    def initial_control(self,N,m=1):

        return np.zeros(self.power+m)


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)]


        
        
