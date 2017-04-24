from crank_nicolson_OCP import create_simple_CN_problem
import matplotlib.pyplot as plt
import numpy as np

def check_exact():

    y0 = 3.2
    yT = 1.5
    T = 1
    a = -2

    problem = create_simple_CN_problem(y0,yT,T,a,c=0)

    N = 100000

    ue,t,_ = problem.simple_problem_exact_solution(N)

    res = problem.solve(N,Lbfgs_options={'jtol':1e-10})
    
    print problem.Functional(ue,N),problem.Functional(res.x,N)
    
    u0 = res.x[-1]

    u_test = lambda x : u0*np.exp(a*(T-t))
    print max(abs(res.x[1:-1]-ue[1:-1]))
    plt.plot(t,ue,'--')
    plt.plot(t,res.x)
    #plt.plot(t,u_test(t),'.')
    plt.show()

if __name__ =='__main__':
    check_exact()

