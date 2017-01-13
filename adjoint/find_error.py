import numpy as np
from optimalContolProblem import OptimalControlProblem,Problem1
from ODE_pararealOCP import SimplePpcProblem,PararealOCP
from scipy.integrate import trapz
import matplotlib.pyplot as plt
def non_lin_problem(y0,yT,T,a):
    
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + 0.5*(y-yT)**2

    def grad_J(u,p,dt):
            
        grad=dt*(u+p)
        #grad[0] = 0.5*u[0]
        #grad[-1] = 0.5*grad[-1]
        return grad
    


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    return problem

def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def also_in_simple():


    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)
    
    N =1000
    m =20
    t = np.linspace(0,T,N+1)
    res=problem.solve(N)
    print
    res2 = problem.penalty_solve(N,m,[1,10,20,50,100,120])#,10000])
    
    opt = {'scale_factor':1,'mem_lim':10,'scale_hessian':True}
    res3 = problem.PPCLBFGSsolve(N,m,[1,100,1000,2000,10000,100000,1000000],options=opt,scale=True)
    for i in range(len(res2)):
        err= l2_diff_norm(res['control'].array(),res2[i]['control'].array()[:N+1],t)
        print err
    print
    for i in range(len(res3)):
        err= l2_diff_norm(res['control'].array(),res3[i]['control'].array()[:N+1],t)
        print err

def check_gather():

    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)

    N = 100
    m = 2
    mu = 1
    u = np.zeros(N+m)

    l,L = problem.adjoint_penalty_solver(u,N,m,mu)


    y,Y = problem.ODE_penalty_solver(u,N,m)
    print l[0][-10:]
    print l[1][:10]
    print L[N/2-10:N/2+10]

    print len(l[0]),len(l[1]),len(L)
    print
    print y[0][-10:]
    print y[1][:10]
    print Y[N/2-10:N/2+10]
    print len(y[0]),len(y[1]),len(Y)
def finite_diff(J,eps,u):
        
    g = np.zeros(len(u))
    for i in range(len(u)):
        Eps = np.zeros(len(u))
        Eps[i] = eps
        g[i] = (J(u+eps) -J(u))/eps

    return g
def finite_difference_grad():
    y0=1
    yT=-10
    T=1
    a=1

    problem = non_lin_problem(y0,yT,T,a)


    N =100

    m = 3
    my=1
    dt = float(T)/N

    J,grad_J = problem.generate_reduced_penalty(dt,N,m,my)

    
    u= np.linspace(0,T,N+m)
    
    eps =0.00000000001
    grad1 = grad_J(u)
    grad2 = finite_diff(J,eps,u)
    
    grad2[:N+1]=dt*grad2[:N+1]
    print grad2
    plt.plot(grad1[:N+1],'r--')
    plt.plot(grad2[:N+1])
    plt.show()

if __name__=='__main__':
    #also_in_simple()
    #check_gather()
    finite_difference_grad()
