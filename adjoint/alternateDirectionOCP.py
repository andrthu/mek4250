from optimalContolProblem import OptimalControlProblem, Problem1
from parallelOCP import interval_partition,v_comm_numbers,u_part
import numpy as np

class AlternateDirectionOCP(OptimalControlProblem):

    def __init__(self,y0,yT,T,J,grad_J,decopled_J,decopled_grad,Lbfgs_options=None,options=None,implicit=True):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,Lbfgs_options,options,implicit)


        self.decopled_J = decopled_J
        self.decopled_grad = decopled_grad


    def partition_control(self,u,N,m):

        new_u = []
        ss,sl,_,_ = v_comm_numbers(N+1,m)
        for i in range(m):

            ui=interval_partition(N+1,m,i)
            ui[:] = u.copy()[ss[i]:ss[i]+sl[i]]
            new_u.append(ui)

        return new_u


class SimpleADOCP(AlternateDirectionOCP):

    def __init__(self,y0,yT,T,a,J,grad_J,decopled_J,decopled_grad,options=None):

        AlternateDirectionOCP.__init__(self,y0,yT,T,J,grad_J,decopled_J,decopled_grad,options)

        self.a = a

    
    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i]+dt*u[j+1])/(1.-dt*a)


    

    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return l[-(i+1)]/(1.-dt*a)

def create_simple_ADOCP_problem(y0,yT,T,a):


    
    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))
        dt = t[1]-t[0]
        I = dt*np.sum(u[1:]**2)#trapz((u)**2,t)

        return 0.5*I + (1./2)*(y-yT)**2

    def grad_J(u,p,dt):
        t = np.linspace(0,T,len(u))
        grad = np.zeros(len(u))
        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0
        #grad[-1] =  dt*p[-2]+dt*(u[-1])
        return grad


    def decomp_J(u,y,lam,mu,dt):

        val = dt*np.sum(u[1:]**2)
        
        pen = mu*(y-lam)**2
        return 0.5*(val+pen)

    def decomp_grad(u,p,dt):

        grad = np.zeros(len(p))

        grad[1:] = dt*(u[1:]+p[:-1])
        grad[0] = 0
        return grad

    
    problem = SimpleADOCP(y0,yT,T,a,J,grad_J,decomp_J,decomp_grad)

    return problem
    


    
def test_stuff():

    y0=1
    a = 1
    yT=1
    T=1

    problem = create_simple_ADOCP_problem(y0,yT,T,a)
    
    N = 1000
    m = 10
    u = np.zeros(N+m)+1

    dt = T/float(N)
    mu = 10
    u2=problem.partition_control(u[:N+1],N,m)

    y,Y = problem.ODE_penalty_solver(u,N,m)
    
    s=0
    for i in range(m-1):
        
        val = problem.decopled_J(u2[i],y[i][-1],u[N+1+i],mu,dt)
        s+=val
        print val
    val = problem.decopled_J(u2[-1],y[-1][-1],yT,1,dt)
    
    print val

    s+=val
    s2 = problem.Penalty_Functional(u,N,m,mu)
    print s,s2,s-s2
    
    p,P = problem.adjoint_penalty_solver(u,N,m,mu)
    
    grads=[]
    
    for i in range(m):
        grads.append(problem.decopled_grad(u2[i],p[i],dt))

    #print grads
    grad_gather = problem.implicit_gather(grads,N,m)
    grad = problem.Penalty_Gradient(u,N,m,mu)
    
    print max(grad_gather-grad[:N+1])

if __name__=='__main__':
    test_stuff()
