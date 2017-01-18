import numpy as np
from optimalContolProblem import OptimalControlProblem,Problem1
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from scipy.integrate import trapz
import matplotlib.pyplot as plt
import pandas as pd

def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def l2_norm(u,t):
    return np.sqrt(trapz((u)**2,t))

def non_lin_problem(y0,yT,T,a):
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
            
        grad=dt*(u+p)
        #grad[0] = dt*0.5*u[0]
        #grad[-1] = dt*(0.5*u[-1]+p[-1])#grad[-1]
        return grad
    


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    return problem

def test(N=1000):

    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)

    
    
    seq_res=problem.solve(N,Lbfgs_options={'jtol':1e-10})

    opt = {'jtol':0,'maxiter':60}

    m = [2,4,8,16,32,64]
    t = np.linspace(0,T,N+1)
    mu_list = [1e1,1e2,1e3,1e4,1e5,1e6,1e7,1e8,1e9,1e10,1e13]
    table={}
    seq_norm = l2_norm(seq_res.x,t)
    for i in range(len(m)):
        res = problem.PPCLBFGSsolve(N,m[i],mu_list,options=opt)

        error = []
        for j in range(len(res)):
            err = l2_diff_norm(res[j].x[:N+1],seq_res.x,t)/seq_norm
            
            error.append(err)

        table.update({m[i]:error})

    data = pd.DataFrame(table,index=mu_list)
        
    print data
    print seq_norm
if __name__ == '__main__':
    test()
