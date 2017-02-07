from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz


class PolynomialControl(Problem1):

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):

        
        Problem1.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power
        self.powers = np.linspace(0,power,power+1)

    def polynomial(self,coef,t):
        
        p = 0
        for i in range(len(coef)):
            p += coef[i]*t**self.powers[i]

        return p       

        
        
    def initial_control(self,N,m=1):

        return np.zeros(self.power+m)


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        t = np.linspace(0,self.T,int(self.T/dt)+1)
        p = self.polynomial(u,t)
        return (y[i] +dt*p[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)]



    def Functional(self,u,N):
        """
        Reduced functional, that only depend on control u 
        """

        return self.J(u,self.ODE_solver(u,N)[-1],self.yT,self.T,N,self.polynomial)       
        

    def Penalty_Functional(self,u,N,m,my):
        """
        Reduced functional, that only depend on control u. Also adds
        penalty terms
        """
        y,Y = self.ODE_penalty_solver(u,N,m)
        Nc = len(u) -m
        J_val = self.J(u[:Nc+1],y[-1][-1],self.yT,self.T,N,self.polynomial)

        penalty = 0

        for i in range(m-1):
            penalty = penalty + my*((y[i][-1]-u[-m+1+i])**2)
        #print penalty
        return J_val + 0.5*penalty

    def Gradient(self,u,N):
        l = self.adjoint_solver(u,N)
        dt = float(self.T)/N
        return self.grad_J(u,l,self.T,N,self.polynomial)

    def Penalty_Gradient(self,u,N,m,mu):

        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
            
        g[:Nc+1]=self.grad_J(u[:Nc+1],L,self.T,N,self.polynomial)

        for j in range(m-1):
            g[Nc+1+j]= l[j+1][0] - l[j][-1]
                    
        return g
        
        
def create_poly_problem():

    T = 1
    a = 1
    y0 = 1
    yT = 1
    
    power = 2

    N = 100

    def J(u,y,yT,T,N,poly):
        val = 0.5*(y-yT)**2
        t = np.linspace(0,T,N+1)
        p = poly(u,t)
        I = 0.5*trapz(p**2,t)
        return val + I

    def grad_J(u,l,T,N,poly):
        t = np.linspace(0,T,N+1)
        g = np.zeros(len(u))
        dt = float(T)/N
        for i in range(len(g)):
            g[i] = dt*np.sum((t**i)*l) + trapz((t**i)*poly(u,t),t)
        #print g
        return g

    problem = PolynomialControl(y0,yT,T,a,power,J,grad_J)
    
    res = problem.solve(N,Lbfgs_options={'ignore xtol':True})
    print res.x
    res2 = problem.penalty_solve(N,2,[1,10,100,10000])
    print res2[-1].x
    import matplotlib.pyplot as plt
    t = np.linspace(0,T,N+1)
    plt.plot(t,problem.polynomial(res.x,t))
    plt.plot(t,problem.polynomial(res2[-1].x[:power+1],t))
    plt.show()
if __name__ == '__main__':
    
    create_poly_problem()
    
    
