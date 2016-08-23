from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *


def make_sin_functional(Time,power=2):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u-np.sin(np.pi*t))**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        t = np.linspace(0,Time,len(u))

        return dt*((u-np.sin(np.pi*t))+p)
        


    return J, grad_J


def test_sin_solution():

    y0 = 1    
    T  = 1
    a  = 1
    K = 1./(a*(1+(np.pi/a)**2))
    K2 = np.pi/a
    #yT = K*(-K2*np.cos(np.pi*T)-np.sin(np.pi*T)) + (y0+K2*K)*np.exp(a*T)
    yT = 1
    J,grad_J = make_sin_functional(T)
    
    N = 500
    problem = Problem1(y0,yT,T,a,J,grad_J)

    res = problem.scipy_solver(N,disp=False)
    res2 = problem.scipy_penalty_solve(N,10,[1000])
    import matplotlib.pyplot as plt
    t = np.linspace(0,T,N+1)
    
    plt.plot(t,res.x)#-np.sin(np.pi*t))
    plt.plot(t,res2.x[:N+1])#-np.sin(np.pi*t))
    #plt.plot(t,np.sin(np.pi*t))
    plt.show()

if __name__ == "__main__":

    test_sin_solution()

    
