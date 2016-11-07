from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem
import time


class PararealOCP(OptimalControlProblem):


    def __init__(self,y0,yT,T,J,grad_J,options=None):
        
        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options=options)
        
        self.end_diff = None
        
    def initial_adjoint(self,y):
        """
        Initial condition for adjoint equation. Depends on the Functional
        """
        self.end_diff = y - self.yT
        return y - self.yT

    def adjoint_step(self,omega,dT,step=1):
        
        step_dt = dT/float(step)

        v = np.zeros(step+1)
        v[-1] = omega
        for i in range(step):
            v[-(i+2)] = self.adjoint_update(v,None,i,step_dt)

        return v[0]

    def ODE_step(self,omega,dT,step=1):
        step_dt = dT/float(step)

        v = np.zeros(step+1)
        v[0] = omega
        for i in range(step):
            v[i+1] = self.ODE_update(v,np.zeros(step+1),i,0,step_dt)
        return v[-1]

    def PC_maker2(self,N,m,step=1):

        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x.copy()[N+1:]
            
            dT = self.T/m
            #S[-1] = self.end_diff
            #S[0] = self.y0

            for i in range(0,m):
                S[-(i+1)] = S[-(i+1)] + self.adjoint_step(S[-(i+1)],dT,step=step)

            #print S
            #time.sleep(1)
            for i in range(1,m+1):
                S[i] = S[i] + self.ODE_step(S[i-1],dT,step=step)

            x[N+1:]=S.copy()[1:-1]
            return x
        return pc

    def adjoint_propogator(self,m,delta0,S):

        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[-1] = delta0

        for i in range(m):

            delta[-(i+2)]=self.adjoint_propogator_update(delta,S,i,dT)

        return delta

    

    def ODE_propogator(self,m,delta0,S):
        T = self.T
        dT = float(T)/m

        delta = np.zeros(m+1)

        delta[0] = delta0

        for i in range(m):
            delta[i+1] = self.ODE_update(delta,S/dT,i,i-1,dT)

        return delta

    def PC_maker(self,N,m):

        
        def pc(x):
            S = np.zeros(m+1)
            S[1:-1] = x[N+1:].copy()
            #S[-1] = self.end_diff
            delta =self.adjoint_propogator(m,0,S)

            for i in range(1,len(S)-1):
                S[i] = S[i]+delta[i]

            #S2 = np.zeros(m)
            #S2[1:]=S[:-1].copy()

            delta2 = self.ODE_propogator(m,0,S)

            for i in range(len(S)-1):
                S[i+1] = S[i+1]+delta2[i+1]
            
            x[N+1:]=S[1:-1]
            
            return x
        
        #def pc(x):
            #return x
        return pc

    def PPCSDsolve(self,N,m,my_list,x0=None,options=None):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        
        result = []
        PPC = self.PC_maker2(N,m,step=10)
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_SD_options(options)
            SDopt = self.SD_options

            Solver = PPCSteepestDecent(J,grad_J,x0.copy(),PPC,
                                       decomp=m,options=SDopt)
            res = Solver.solve()
            x0=res.x
            result.append(res)
        if len(result)==1:
            y,Y = self.ODE_penalty_solver(x0,N,m)
            import matplotlib.pyplot as plt
            plt.plot(Y)
            plt.show()
            return res
        else:
            return result


    def adjoint_propogator_update(self,l,rhs,i,dt):
        raise NotImplementedError,'not implemented'

class SimplePpcProblem(PararealOCP):
    """
    optimal control with ODE y'=ay+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        PararealOCP.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return l[-(i+1)]/(1.-dt*a) 


    def adjoint_propogator_update(self,l,rhs,i,dt):
        a = self.a
        return (l[-(i+1)]+rhs[-(i+1)])/(1-dt*a)



def find_gradient():

    import matplotlib.pyplot as plt
    y0 = 1
    yT = 1
    T  = 1
    a  = 1
    N = 1000
    m = 10
    dt =float(T)/N

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        #grad =np.zeros(len(u))
        #grad[0] = 0.5*u[0]
        #grad[1:-1] = u[1:-1]+p[1:-1]
        #grad[-1] = 0.5*u[-1]+p[-1]
        #return dt*grad
        return dt*(u+p)

    Problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)

    red_J,red_grad_J = Problem.generate_reduced_penalty(dt,N,m,10)

    x = np.zeros(N+m)

    grad = red_grad_J(x)

    print grad

    lam = np.zeros(m+1)
    

    lam[1:-1] = grad[N+1:]
    plt.plot(np.linspace(0,T,m+1),lam)
    plt.show()

    delta = Problem.adjoint_propogator(m,0,lam)

    for i in range(1,len(lam)-1):
        lam[i] = lam[i]+delta[i]

    plot(np.linspace(0,T,m+1),lam,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()


    delta = Problem.ODE_propogator(m,0,lam)
    
    for i in range(len(lam)-1):
        lam[i+1] = lam[i+1]+delta[i+1]

    
    plot(np.linspace(0,T,m+1),lam,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()



    
    

if __name__ == "__main__":
    from matplotlib.pyplot import *
    find_gradient()
    
    y0 = 1
    yT = 10
    T  = 1
    a  = 1

    

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        """
        grad =np.zeros(len(u))
        grad[0] = 0.5*u[0]
        grad[1:-1] = u[1:-1]+p[1:-1]
        grad[-1] = 0.5*u[-1]+p[-1]
        return dt*grad
        """
        return dt*(u+p)


    problem = SimplePpcProblem(y0,yT,T,a,J,grad_J)
    
    N = 1000
    m = 10

    res = problem.PPCSDsolve(N,m,[1])

    res2 = problem.scipy_solver(N,disp=True)
    
    res3 = problem.solve(N,algorithm='my_steepest_decent')
    res4 = problem.penalty_solve(N,m,[500],algorithm='my_lbfgs')

    res5 = problem.scipy_penalty_solve(N,m,[500],disp=True)
    plot(np.linspace(0,T,N+1),res3.x,'--r')
    plot(np.linspace(0,T,N+1),res.x[:N+1],'--b')
    plot(np.linspace(0,T,N+1),res5.x[:N+1])
    plot(np.linspace(0,T,N+1),res4['control'].array()[:N+1])
    plot(np.linspace(0,T,N+1),res2.x)
    show()
    """
    x = res4['control'].array()
    lam = np.zeros(m+1)
    lam[1:-1] = x[N+1:]

    lam[0] = y0
    plot(np.linspace(0,T,m+1),lam)
    
    
    y,Y =problem.ODE_penalty_solver(x,N,m)
    print len(Y)
    plot(np.linspace(0,T,N+1),Y)
    show()
    
    S = np.zeros(m+1)
    S[1:-1]=1

    delta = problem.adjoint_propogator(m,0,S)

    for i in range(len(S)-1):
        S[i] = S[i]+delta[i+1]

    plot(np.linspace(0,T,m+1),S,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()


    delta = problem.ODE_propogator(m,0,S)
    
    for i in range(len(S)-1):
        S[i+1] = S[i+1]+delta[i+1]

    
    plot(np.linspace(0,T,m+1),S,'r-')
    plot(np.linspace(0,T,m+1),delta,'b--')
    show()

    find_gradient()
    """
"""
-(l[-(i+1)]-l[(i+2)])/dT = a*l[-(i+2)] + S[-(i+1)]/dT

l[-(i+2)](1-a*dT) = l[-(i+1)] + S[-(i+1)]

l[-(i+2)] = (l[-(i+1)]+S[-(i+1)])/(1-dT*a) 
"""
