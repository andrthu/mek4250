import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taylorTest import lin_problem


def unstable():

    a = -0.02
    T = 100

    y0=3
    yT= 100

    problem,_ = lin_problem(y0,yT,T,a,implicit=True)
    
    N = 10000
    m = 50
    res = problem.PPCLBFGSsolve(N,m,[100,1000],options={'jtol':1e-5})[-1]
    #res = problem.penalty_solve(N,m,[100,1000],Lbfgs_options={'jtol':1e-6})[-1]
    res2 = problem.solve(N,Lbfgs_options={'jtol':1e-10})
    print res2.counter(),res.counter(),max(abs(res2.x-res.x[:N+1]))
    Y  =problem.ODE_solver(np.zeros(m+1),m)
    t = np.linspace(0,T,N+1)
    #Y = problem.ODE_solver(u,N)
    plt.plot(t,res.x[:N+1])
    plt.plot(t,res2.x,'--')
    #plt.plot(np.linspace(0,T,m+1),Y)
    plt.show()
if __name__ == '__main__':
    unstable()
