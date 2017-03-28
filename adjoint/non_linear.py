from  optimalContolProblem import *

import numpy as np
from scipy.integrate import trapz



class Explicit_quadratic(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        #return (y[i] +dt*u[j+1])/(1.-dt*a)
        
        
        return y[i] - dt*(a*y[i]**2 - u[j])


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]
        #return (1-dt*a)*l[-(i+1)]
        return (1-a*2*dt*y[-(i+1)])*l[-(i+1)]


    def Penalty_Gradient(self,u,N,m,mu):

        l,L = self.adjoint_penalty_solver(u,N,m,mu)
        dt = float(self.T)/N
        Nc = len(u) - m
        g = np.zeros(len(u))
        a = self.a
        g[:Nc+1]=self.grad_J(u[:Nc+1],L,dt)

        for j in range(m-1):
            g[Nc+1+j]= (l[j+1][0]*(1-dt*2*a*u[Nc+1+j]) - l[j][-1])
                    
        return g


class Explicit_sine(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options,implicit=False)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        
        #return y[i] +dt*(a*y[i] +u[j])
        return y[i] + dt*(a*np.sin(y[i]) + u[j])


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        #return (1+dt*a)*l[-(i+1)]

        return (1+dt*a*np.cos(y[-(i+1)]))*l[-(i+1)]
    
class ExplicitNonLinear(OptimalControlProblem):

    def __init__(self,y0,yT,T,F,DF,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.F  = F
        self.DF = DF

    def ODE_update(self,y,u,i,j,dt):
        F = self.F
        
        
        return y[i] + dt*(F(y[i]) + u[j])


    def adjoint_update(self,l,y,i,dt):
        DF = self.DF
        

        return (1+dt*DF(y[-(i+1)]))*l[-(i+1)]


def test_nonLinear():

    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 2
    T  = 1
    N1 = 2000
    N2 = 5000
    m=10
    a=0.1
    #"""
    F  = lambda x : a*np.exp(-x)*np.cos(x*np.pi)
    
    DF = lambda x : -a*np.exp(-x)*(np.pi*np.sin(np.pi*x)+np.cos(np.pi*x))
    """
    F = lambda x : np.exp(-x**2)
    DF = lambda x : -2*x*np.exp(-x**2)
    """
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = ExplicitNonLinear(y0,yT,T,F,DF,J,grad_J)

    opt = {"mem_lim":40}
    
    res1=problem.plot_solve(N1,opt=opt,state= True)
    #res2=problem.solve(N2,Lbfgs_options=opt)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(t1,res1['control'])
    #plt.plot(t2,res2['control'],"r--")

    
    try:
        res3=problem.penalty_solve(N1,m,[0.5*N1],Lbfgs_options=opt)
        print res1['iteration'],res3['iteration']
        plt.plot(t1,res3['control'][:N1+1])
    except:
        #print res1['iteration'],res2['iteration']
        print res1['iteration']
    plt.show()
    
def test_sine():

    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 2
    T  = 3
    a  = 0.1
    N1 = 1000
    N2 = 2000
    m=2
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        grad = np.zeros(len(u))
        grad[:-1] = dt*(u[:-1]+p[1:])
        grad[0] = 0.5*dt*(u[0])+dt*p[1] 
        grad[-1] = 0.5*dt*(u[-1]) 
        return grad
        #return dt*(u+p)


    problem = Explicit_sine(y0,yT,T,a,J,grad_J)

    opt = {"mem_lim":10,'jtol':1e-7}
    res1=problem.solve(N1,Lbfgs_options=opt)
    res2=problem.solve(N2,Lbfgs_options=opt)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(t1,res1['control'])
    plt.plot(t2,res2['control'],"r--")

    
    try:
        res3=problem.penalty_solve(N1,m,[1,N1],Lbfgs_options={'jtol':1e-4})[-1]
        #print res1['iteration'],res2['iteration'],res3['iteration']
        print res3.counter(),res1.counter()
        print
        print "number of iterations for m=%d and N=%d: %d"%(1,N1,res1['iteration'])
        print "number of iterations for m=%d and N=%d: %d"%(1,N2,res2['iteration'])
        print "number of iterations for m=%d and N=%d: %d"%(m,N1,res3['iteration'])
        print
        plt.plot(t1,res3['control'][:N1+1])
        plt.legend(['m=1,N='+str(N1),'m=1,N='+str(N2),'m='+str(m)+',N='+str(N1)])
    except:
        print 'lol'
    plt.show()
    
def test_quad():
    
    import matplotlib.pyplot as plt

    y0 = 1
    yT = 5
    T  = 1
    a  = 1
    N1 = 1000
    N2 = 2000
    m=30
    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = Explicit_quadratic(y0,yT,T,a,J,grad_J)

    opt = {"mem_lim":80}
    res1=problem.solve(N1,Lbfgs_options=opt)
    res2=problem.solve(N2,Lbfgs_options=opt)
    
    t1 = np.linspace(0,T,N1+1)
    t2 = np.linspace(0,T,N2+1)
    plt.figure()
    plt.plot(t1,res1['control'])
    plt.plot(t2,res2['control'],"r--")

    
    
    res3=problem.penalty_solve(N1,m,[0.5*N1],Lbfgs_options=opt)
    #print res1['iteration'],res2['iteration'],res3['iteration']
    print
    print "number of iterations for m=%d and N=%d: %d"%(1,N1,res1['iteration'])
    print "number of iterations for m=%d and N=%d: %d"%(1,N2,res2['iteration'])
    print "number of iterations for m=%d and N=%d: %d"%(m,N1,res3['iteration'])
    print
    plt.plot(t1,res3['control'][:N1+1])
    plt.legend(['m=1,N='+str(N1),'m=1,N='+str(N2),'m='+str(m)+',N='+str(N1)],loc=2)
    plt.show()
    
if __name__ == "__main__":

    #test_quad()
    test_sine()
    #test_nonLinear()
    
"""
terminal> python non_linear (test_quad())

number of iterations for m=1 and N=1000: 11
number of iterations for m=1 and N=2000: 11
number of iterations for m=30 and N=1000: 53


"""
