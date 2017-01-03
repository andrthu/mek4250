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


class GeneralPowerEndTermPCP(SimplePpcProblem):

    """
    class for the opti-problem:
    J(u,y) = 0.5*||u||**2 + 1/p*(y(T)-yT)**p
    with y' = ay + u
    """

    def __init__(self,y0,yT,T,a,power,J,grad_J,options=None):
        SimplePpcProblem.__init__(self,y0,yT,T,a,J,grad_J,options)
        self.power = power
    
        def J_func(u,y,yT,T):
            return J(u,y,yT,T,self.power)
        
        self.J = J_func

    def initial_adjoint(self,y):
        
        p = self.power
        return (y - self.yT)**(p-1)


def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def test1():

    y0 = 1
    yT = 30
    T  = 1
    a  = 1
    N  = 500
    m  = 10

    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        
        return dt*(u+p)


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    
    res1 = problem.solve(N)

    res2 = problem.PPCLBFGSsolve(N,m,[10])
    res3 = problem.penalty_solve(N,m,[10])
    print res2.niter,res3['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array(),'r--')
    plt.plot(t,res2.x[:N+1])
    plt.plot(t,res3['control'].array()[:N+1])
    plt.show()


def test2():
    y0 = 1
    yT = 10
    T  = 1
    a  = 1
    N  = 500
    m  = 10
    p = 4
    

    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        
        return dt*(u+p)


    problem = GeneralPowerEndTermPCP(y0,yT,T,a,p,J,grad_J)

    res1 = problem.solve(N)
    res2 = problem.PPCLBFGSsolve(N,m,[100])
    res3 = problem.penalty_solve(N,m,[100])
    print res1['iteration'],res2.niter,res3['iteration']
    t = np.linspace(0,T,N+1)
    plt.plot(t,res1['control'].array(),'r--')
    plt.plot(t,res2.x[:N+1])
    plt.show()

def non_lin_problem(y0,yT,T,a,p):

    def J(u,y,yT,T,power):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*I + (1./power)*(y-yT)**power

    def grad_J(u,p,dt):
        
        return dt*(u+p)


    problem = GeneralPowerEndTermPCP(y0,yT,T,a,p,J,grad_J)

    return problem


def compare_pc_and_nonpc_for_different_m():

    y0 = 3.2
    yT = 1.5
    T  = 1
    a  = 0.9
    p = 4


    
    problem = non_lin_problem(y0,yT,T,a,p)
    N = 800
    M = [1,2,4,8,16,32,64]
    
    res1 = problem.solve(N)

    mu = 5

    table = {'pc itr'          : ['--'],
             'non-pc itr'      : ['--'],
             'pc err'          : ['--'],
             'non-pc err'      : ['--'],
             'non_penalty itr' : [res1['iteration']],}


    t = np.linspace(0,T,N+1)

    res2 = []
    res3 = []
    opt = {'scale_factor':1,'mem_lim':10,'scale_hessian':True}
    for m in M[1:]:

        pc_res = problem.PPCLBFGSsolve(N,m,[m*mu])
        nonpc_res = problem.penalty_solve(N,m,[m*mu],Lbfgs_options=opt,scale=True)

        res2.append(pc_res)
        res3.append(nonpc_res)
    
        err1 = l2_diff_norm(res1['control'].array(),pc_res.x[:N+1],t)
        err2 = l2_diff_norm(res1['control'].array(),nonpc_res['control'].array()[:N+1],t)

        table['pc itr'].append(pc_res.niter)
        table['non-pc itr'].append(nonpc_res['iteration'])
        table['pc err'].append(err1)
        table['non-pc err'].append(err2)
        table['non_penalty itr'].append('--')


    data = pd.DataFrame(table,index=M)
    print data

    plt.figure()
    plt.plot(t,res1['control'].array(),'r--')
    for i in range(len(res2)):
        plt.plot(t,res2[i].x[:N+1])
    plt.legend(M,loc=4)
    plt.show()
    
    plt.figure()
    plt.plot(t,res1['control'].array(),'r--')
    for i in range(len(res2)):
        plt.plot(t,res3[i]['control'].array()[:N+1])
    plt.legend(M,loc=4)
    plt.show()
                

    

if __name__ == '__main__':
    #test1()
    #test2()
    compare_pc_and_nonpc_for_different_m()
