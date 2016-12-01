from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
import matplotlib.pyplot as plt


def test1():
    
    T=1
    y0=1
    a=1
    alpha=1
    yT=1

    N = 800
    m = 10
    mu=100

    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    


    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)

    res1=problem.penalty_solve(N,m,[mu])
    res2=problem.penalty_solve(N,m,[mu],scale=True)
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array()[:N+1])
    plt.plot(t,res2['control'].array()[:N+1],'r--')
    plt.show()
    plt.plot(res1['control'].array()[N+1:])
    plt.plot(res2['control'].array()[N+1:])
    plt.show()
test1()

