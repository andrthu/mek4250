from ODE_pararealOCP import PararealOCP
import numpy as np
from scipy.integrate import trapz
from taylorTest import general_taylor_test,lin_problem

class CrankNicolsonOCP(PararealOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):
 
        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=True)

        self.a = a

        


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        return (y[i]*(1+0.5*dt*a)+0.5*dt*(u[j]+u[j+1]))/(1-0.5*dt*a)
        
    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return (1+0.5*dt*a)*l[-(i+1)]/(1-0.5*dt*a)

    def implicit_gather(self,l,N,m):
        L=np.zeros(N+1)
        #"""
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            if i!=0:
                L[start] = 0.5*(l[i][0]+l[i-1][-1])
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
        return L
class CrankNicolsonStateIntOCP(PararealOCP):
    def __init__(self,y0,yT,T,a,J,grad_J,options=None):
 
        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options,implicit=True)

        self.a = a

    def implicit_gather(self,l,N,m):
        L=np.zeros(N+1)
        #"""
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            if i!=0:
                L[start] = 0.5*(l[i][0]+l[i-1][-1])
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
        return L


def create_simple_CN_problem(y0,yT,T,a):
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz((u)**2,t)

        return 0.5*I + (1./2)*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:-1] = dt*(u[1:-1]+p[1:-1])
        grad[0] = 0.5*dt*(u[0]+p[0]) 
        grad[-1] = 0.5*dt*(u[-1]) + 0.5*dt*p[-1]
        return grad



    problem = CrankNicolsonOCP(y0,yT,T,a,J,grad_J)
    return problem

def test_CN():

    a = 1
    y0=1
    T=1
    yT=1

    problem = create_simple_CN_problem(y0,yT,T,a)
    problem2,_ = lin_problem(y0,yT,T,a)
    N = 10000
    opt = {'jtol':0,'maxiter':50}
    res = problem.solve(N,Lbfgs_options=opt )
    res2 = problem2.solve(N,Lbfgs_options=opt)

    res3 = problem.penalty_solve(N,10,[N,N**2],Lbfgs_options=opt)[-1]
    res4 = problem.PPCLBFGSsolve(N,10,[N,N**2],options=opt)[-1]

    import matplotlib.pyplot as plt
    print res.counter(),res2.counter(),res3.counter(),res4.counter()
    plt.plot(res.x,'--')
    plt.plot(res2.x)
    plt.plot(res3.x[:N+1])
    plt.plot(res4.x[:N+1],'o')
    plt.show()

def taylorTestCN():
    a = 1
    y0=1
    T=1
    yT=1

    problem = create_simple_CN_problem(y0,yT,T,a)
    
    general_taylor_test(problem)

if __name__=='__main__':

    test_CN()
    #taylorTestCN()
