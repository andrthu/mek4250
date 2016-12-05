from optimalContolProblem import *
from non_linear import *
import numpy as np
from scipy.integrate import trapz
from scipy import linalg
from cubicYfunc import *
import matplotlib.pyplot as plt
import pandas as pd


def test1():
    
    T=1
    y0=1
    a=1
    alpha=0.2
    yT=10

    N = 800
    m = 20
    mu=1

    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    


    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)

    res1=problem.penalty_solve(N,m,[mu])
    res2=problem.penalty_solve(N,m,[mu],scale=True)
    print res1['iteration'],res2['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array()[:N+1])
    plt.plot(t,res2['control'].array()[:N+1],'r--')
    plt.show()
    plt.plot(res1['control'].array()[N+1:])
    plt.plot(res2['control'].array()[N+1:])
    plt.show()


def test2():
    y0 = 3
    yT = 0
    T  = 1.
    a  = 1.3
    P  = 4
    N=700
    m = 10
    mu = 10
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
    res=problem.penalty_solve(N,m,[mu],scale=True)
    res2=problem.penalty_solve(N,m,[mu],scale=False)
    print(res['iteration'],res2['iteration'])

    t = np.linspace(0,T,N+1)
    plt.plot(t,res['control'].array()[:N+1],'r--')
    plt.plot(t,res2['control'].array()[:N+1])
    plt.show()
def test3():

    T=1
    y0=1
    a=1
    alpha=0.2
    yT=0

    N = 800
    m = 5
    mu=10

    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    
    mem=[0,1,5]
    table = {'unscaled'         : [],
             'scaled'           : [],
             'scaled hessian'   : [],
             'steepest descent' : []}
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    res3=problem.penalty_solve(N,m,[mu],algorithm='my_steepest_decent',scale=True)
    for i in range(len(mem)):
        problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
        opt1 = {'mem_lim':mem[i],'maxiter':200,'scale_hessian':False}
        opt2 = {'mem_lim':mem[i],'maxiter':200,'scale_hessian':True}
        res1=problem.penalty_solve(N,m,[mu],Lbfgs_options=opt1)
        res2=problem.penalty_solve(N,m,[mu],scale=True,Lbfgs_options=opt2)
        #res3=problem.penalty_solve(N,m,[mu],algorithm='my_steepest_decent',scale=True)
        res4=problem.penalty_solve(N,m,[mu],scale=True,Lbfgs_options=opt1)
        print res1['iteration'],res2['iteration'],res3.niter
        
        table['unscaled'] = res1['iteration']
        table['scaled'] = res4['iteration']
        table['scaled hessian'] = res2['iteration']
        table['steepest descent'] = res3.niter

        t = np.linspace(0,T,N+1)
        plt.plot(t,res1['control'].array()[:N+1])
        plt.plot(t,res2['control'].array()[:N+1],'r--')
        plt.plot(t,res3.x[:N+1])
        #plt.show()
        plt.plot(res1['control'].array()[N+1:])
        plt.plot(res2['control'].array()[N+1:])
        #plt.show()
        
    iter_data = pd.DataFrame(table,index=['mem_lim=0','mem_lim=1','mem_lim=5'])
    print iter_data
    iter_data.to_latex('iter_data.tex')
#test1()
#test2()
test3()
