from ODE_pararealOCP import PararealOCP
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem
import time
import matplotlib.pyplot as plt

class Problem3(PararealOCP):
    """
    optimal control with ODE y=ay'+u
    and J(u)=||u||**2 + alpha*(y(T)-yT)**2
    """

    def __init__(self,y0,yT,T,a,alpha,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a
        self.alpha = alpha
        
        def JJ(u,y,yT,T):
            return J(u,y,yT,T,self.alpha)
        def grad_JJ(u,p,dt):
            return grad_J(u,p,dt,self.alpha)
        self.J=JJ
        self.grad_J = grad_JJ


    def initial_adjoint(self,y):
        return self.alpha*(y - self.yT)

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a)

def test1():

    y0 = 1.2
    a = 0.9
    T = 1.
    yT = 5
    alpha = 0.5
    N = 500
    m = 10
    
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    
    t0=time.time()
    res = problem.solve(N,algorithm='my_steepest_decent')
    t1=time.time()
    normal_time = t1-t0

    mu_val = [1,10,50,75,100,120,175,200]

    t0=time.time()
    res2=problem.PPCSDsolve(N,m,mu_val)
    t1 = time.time()
    penalty_time = t1-t0
    
    

    

    print res.val(),res.niter
    t = np.linspace(0,T,N+1)
    plt.figure()
    plt.plot(t,res.x,'r--')
    leg = []
    leg.append('normal')
    sum_iter = 0
    for i in range(len(res2)):
        print mu_val[i],res2[i].val(),res2[i].niter
        print 'l2 diff:',l2_diff_norm(res.x,res2[i].x[:N+1],t)
        plt.plot(t,res2[i].x[:N+1])
        leg.append('mu='+str(mu_val[i]))
        sum_iter += res2[i].niter
    print sum_iter, sum_iter/float(m), res.niter
    print 'l2 diffrence in control:',l2_diff_norm(res.x,res2[-1].x[:N+1],t)
    print 'normal time:',normal_time,',penalty time:',penalty_time, ',perfect parallel:', penalty_time/m
    plt.legend(leg)
    plt.title('Optimal control')
    plt.xlabel('time')
    plt.ylabel('control')
    #plt.savefig('test1_PC_control.png')
    plt.show()

    plt.figure()
    y1 = problem.ODE_solver(res.x,N)
    plt.plot(t,y1,'r--')
    for i in range(len(res2)):
        _,y = problem.ODE_penalty_solver(res2[i].x,N,m)
        plt.plot(t,y)
    plt.legend(leg,loc=2)
    plt.title('Solution of state equation')
    plt.xlabel('time')
    plt.ylabel('state')
    #plt.savefig('test1_PC_state.png')
    plt.show()

    if False:
        sd_opt = {'maxiter':1000}
        t0 = time.time()
        res3 = problem.penalty_solve(N,m,mu_val,algorithm='my_steepest_decent',
                                     Lbfgs_options=sd_opt)
        t1 = time.time()
        noPC_time = t1-t0

        plt.figure()
        plt.plot(t,res.x,'r--')
        sum_iter = 0
        for i in range(len(res3)):
            print mu_val[i],res3[i].val(),res3[i].niter
            plt.plot(t,res3[i].x[:N+1])
            
            sum_iter += res3[i].niter
        print sum_iter, noPC_time,noPC_time/m
        plt.legend(leg)
        plt.title('Optimal control without preconditioner')
        plt.xlabel('time')
        plt.ylabel('control')

        plt.show()
def l2_diff_norm(u1,u2,t):
    return np.sqrt(trapz((u1-u2)**2,t))

def linf_diff_norm(u1,u2,t):
    return np.max(abs(u1-u2))

def mu_update(mu,num_iter,m,val):

    factor = (1+val*float(m)/num_iter)
    return factor*mu 
def test2():
    
    y0 = 10
    a = 1
    T = 1.
    yT = 13
    alpha = 2
    N = 800
    m = 20
    
    def J(u,y,yT,T,alp):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + alp*(y-yT)**2)

    def grad_J(u,p,dt,alp):
        return dt*(u+p)
    
    problem = Problem3(y0,yT,T,a,alpha,J,grad_J)
    res = problem.solve(N,algorithm='my_steepest_decent')
    
    print res.val(),res.niter
    t = np.linspace(0,T,N+1)
    plt.plot(t,res.x)
    plt.show()
    
    x = np.zeros(N+m)
    mu = 1
    sum_iter = 0
    while l2_diff_norm(x[:N+1],res.x,t)>1:
        res2 = problem.PPCSDsolve(N,m,[mu],x0=x)
        x = res2.x.copy()
        mu = mu_update(mu,res2.niter,m,20)
        print mu
        plt.plot(t,res2.x[:N+1])
        plt.plot(t,res.x,'r--')
        sum_iter += res2.niter
        print l2_diff_norm(x[:N+1],res.x,t)
        plt.show()
    print sum_iter,sum_iter/float(m),res.niter
    return

if __name__ == '__main__':
    test1()
    #test2()

