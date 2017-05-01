import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from taylorTest import lin_problem


def unstable():

    a = -0.2
    T = 100

    y0=3
    yT= 10

    problem,_ = lin_problem(y0,yT,T,a,implicit=False)
    
    N = 10

    u = np.zeros(N+1)
    t = np.linspace(0,T,N+1)
    Y = problem.ODE_solver(u,N)
    plt.plot(t,Y)
    plt.show()
if __name__ == '__main__':
    unstable()
