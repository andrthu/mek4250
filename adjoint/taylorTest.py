import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.splitLbfgs import SplitLbfgs
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
from optimalContolProblem import OptimalControlProblem,Problem1


from test_LbfgsPPC import GeneralPowerEndTermPCP,non_lin_problem
from runge_kutta_OCP import RungeKuttaProblem

def lin_problem(y0,yT,T,a):
    
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + (1./2)*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0.5*dt*(u[0]) 
        grad[-1] = 0.5*dt*(u[-1]) + dt*p[-2]
        return grad



    problem1 = RungeKuttaProblem(y0,yT,T,a,J,grad_J)
    problem2 = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    problem2 = Problem1(y0,yT,T,a,J,grad_J)
    return problem2,problem1


def taylor_test_non_penalty():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(10):

        print J(u+h/(10**i))-J(u)


    def grad_J(x):
        l = problem.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)
    def grad_J2(x):
        l = problem2.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)
    print

    for i in range(10):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(10):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    #grad2 = grad_J2(u)
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    plt.show()

def taylor_penalty_test():
    y0 = 3.2
    yT = 100.5
    T  = 1
    a  = 10.9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    #problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    m = 2
    my = 1
    

    J,grad_J = problem.generate_reduced_penalty(dt,N,m,my)
    h = 100*np.random.random(N+m)
    u = np.zeros(N+m) 
    for i in range(10):

        print abs(J(u+h/(10**i))-J(u))


    print

    for i in range(10):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(10):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    
    plt.show()
    
def quad_end():
    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 1.9
    p = 2
    c =0
    
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    my =10
    m=6
    h = 100*np.random.random(N+1)
    
    

    J = lambda u: problem.Functional(u,N)
    u = np.zeros(N+1) +1
    for i in range(10):

        print J(u+h/(10**i))-J(u)


    def grad_J(x):
        l = problem.adjoint_solver(u,N)
        return problem.grad_J(u,l,dt)

    print

    for i in range(9):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(9):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    #grad2 = grad_J2(u)
    #plt.plot(grad)
    #plt.plot(grad_fd,'r--')
    #plt.plot(grad2)
    #plt.show()
    
    print
    print 'Penalty:'
    J,grad_J = problem.generate_reduced_penalty(dt,N,m,my)
    h = 100*np.random.random(N+m)
    u = np.zeros(N+m) 
    for i in range(10):

        print abs(J(u+h/(10**i))-J(u))


    print

    for i in range(10):
        eps = 1./(10**i)
        print abs(J(u+h*eps) - J(u) - eps*h.dot(grad_J(u)))
    print

    for i in range(10):
        eps = 1./(10**i)
        grad_fd = finite_diff(J,u,eps)
        grad = grad_J(u)
        print max(abs(grad_fd[:]-grad[:]))
    
    import matplotlib.pyplot as plt
    
    plt.plot(grad)
    plt.plot(grad_fd,'r--')
    
    plt.show()

    y,Y = problem.ODE_penalty_solver(u,N,m)
    plt.plot(Y)
    plt.show()
    
    p,P = problem.adjoint_penalty_solver(u,N,m,my)
    plt.plot(P)
    plt.show()
def finite_diff(J,u,epsilon):
    grad = np.zeros(len(u))

    for i in range(len(u)):
        Eps = np.zeros(len(u))
        Eps[i] = epsilon
        
        grad[i] = (J(u+Eps)-J(u))/epsilon

    return grad


def augemted_test():


    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 2
    c =0.5

    problem,problem2 = lin_problem(y0,yT,T,a)
    problem = non_lin_problem(y0,yT,T,a,p,c=c)
    N = 100
    dt = 1./(N)
    
    h = 100*np.random.random(N+1)


    



if __name__ == '__main__':
    #taylor_test_non_penalty()
    taylor_penalty_test()
    #quad_end()


"""
J(u+eh) = J(u) + O(e)

J(u+eh) = J(u) + eh*grad_J(u) + O(e**2)

"""
