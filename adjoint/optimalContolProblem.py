from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np

class OptimalControlProblem():

    def __init__(self,y0,yT,T,J,grad_J,Lbfgs_options=None,options=None):
        """
        Initialize OptimalControlProblem

        Valid options are:
        - T: End time 
        - J: Functional depending on u and y and dt
        - options: More options
        """
        self.T      = T
        self.y0     = y0
        self.yT     = yT
        self.J      = J
        self.grad_J = grad_J
        
        self.set_options(options,Lbfgs_options)
        
        self.Vec = self.options['Vector']

    def set_options(self,user_options,user_Lbfgs):
        """
        Method for setting options
        """
        options = self.default_options()
        Lbfgs_options = self.default_Lbfgs_options()
        
        if user_options!=None:
            for key, val in user_options.iteritems():
                options[key]=val

        if user_Lbfgs!=None:
            for key, val in user_Lbfgs.iteritems():
                Lbfgs_options[key]=val

        
        self.options = options
        self.Lbfgs_options = Lbfgs_options

    def default_Lbfgs_options(self):
        
        default = {"jtol"                   : 1e-4,
                   "maxiter"                : 500,
                   "mem_lim"                : 10,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : True,}

        return default

    def default_options(self):

        default = {"Vector" : SimpleVector,
                   "Lbfgs"  : Lbfgs,}
        return default
        
        
    def ODE_update(self,y,u,i,j,dt):
        raise NotImplementedError,'ODE_update not implemented'

    def adjoint_update(self,l,y,i,dt):
        raise NotImplementedError,'adjoint_update not implemented'

    def initial_adjoint(self,y):
        return y - self.yT
    
    def ODE_solver(self,u,N):
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        
        
        
        y = np.zeros(N+1)
        y[0]=y0

        for i in range(N):
            y[i+1] = self.ODE_update(y,u,i,i,dt)

    
        return y

    def ODE_penalty_solver(self,u,N,m):
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        y = partition_func(N+1,m)

        y[0][0]=y0
        for i in range(1,m):
            y[i][0] = u[N+i]

        start=0
        for i in range(m):        
            for j in range(len(y[i])-1):
                y[i][j+1] = self.ODE_update(y[i],u,j,start+j,dt)
            start = start + len(y[i]) - 1
            

        Y = np.zeros(N+1)
        start = 0
        for i in range(m):
            Y[start:start+len(y[i])-1] = y[i][:-1]
            start = start + len(y[i]) - 1
        Y[-1] = y[-1][-1]
            
        return y,Y

    def adjoint_solver(self,u,N):
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
              

        y = self.ODE_solver(u,N)

        l=np.zeros(N+1)
        
        l[-1] = self.initial_adjoint(y[-1])
        for i in range(N):
            l[-(i+2)] = self.adjoint_update(l,y,i,dt)
             
        return l
    
    def adjoint_penalty_solver(self,u,N,m,my):

        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        
        l = partition_func(N+1,m)
        y,Y = self.ODE_penalty_solver(u,N,m)

            
        l[-1][-1] = self.initial_adjoint(y[-1][-1])
        for i in range(m-1):
            l[i][-1]=my*(y[i][-1]-u[N+1+i])

        for i in range(m):
            for j in range(len(l[i])-1):
                l[i][-(j+2)] = self.adjoint_update(l[i],y[i],j,dt)
            
        L=np.zeros(N+1)
            
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
            
        return l,L
        
        
    def Functional(self,u,N):

        return self.J(u,self.ODE_solver(u,N)[-1],self.yT,self.T)

    def Penalty_Functional(self,u,N,m,my):

        y,Y = self.ODE_penalty_solver(u,N,m)

        J_val = self.J(u[:N+1],y[-1][-1],self.yT,self.T)

        penalty = 0

        for i in range(m-1):
            penalty = penalty + my*((y[i][-1]-u[N+1+i])**2)
    
        return J_val + 0.5*penalty


    
    
    def solve(self,N,x0=None,Lbfgs_options=None):

        
        dt=float(self.T)/N
        if x0==None:
            x0 = self.Vec(np.zeros(N+1))
            
        def J(u):
            return self.Functional(u,N)

        def grad_J(u):
            l = self.adjoint_solver(u,N)
            return self.grad_J(u,l,dt)

        if Lbfgs_options==None:
            Loptions = self.Lbfgs_options
        else:
            Loptions = self.Lbfgs_options
            for key, val in Lbfgs_options.iteritems():
                Loptions[key]=val
            
        
        solver = Lbfgs(J,grad_J,x0,options=Loptions)

        res = solver.solve()

        return res


    def penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None):


        dt=float(self.T)/N
        if x0==None:
            x0 = self.Vec(np.zeros(N+m))
        x = None
        Result = []
        for i in range(len(my_list)):
        
            def J(u):                
                return self.Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):

                l,L = self.adjoint_penalty_solver(u,N,m,my_list[i])

                g = np.zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)

                for j in range(m-1):
                    g[N+1+j]=l[j+1][0]-l[j][-1]
                    
                return g
            
            if Lbfgs_options==None:
                Loptions = self.Lbfgs_options
            else:
                Loptions = self.Lbfgs_options
                for key, val in Lbfgs_options.iteritems():
                    Loptions[key]=val

            
            solver = Lbfgs(J,grad_J,x0,options=Loptions)
            
            res = solver.solve()
            Result.append(res)
            x0 = res['control'].array()
            
        if len(Result)==1:
            return res
        else:
            return Result

    
    def plot_solve(self,N,opt=None,state=False):
        res = self.solve(N,Lbfgs_options=opt)
        t = np.linspace(0,self.T,N+1)
        u = res['control'].array()

        import matplotlib.pyplot as plt

        if state==False:
            plt.plot(t,u)
            plt.ylabel('control')
            plt.xlabel('time')
        else:
            y = self.ODE_solver(u,N)
            plt.plot(t,u)
            plt.plot(t,y)
            plt.xlabel('time')
            plt.legend(["con","state"])
            
        plt.show()

        return res
        
    def penalty_and_normal_solve(self,N,m,my_list,x0=None,Lbfgs_options=None,
                                 show_plot=False):

        res1 = self.penalty_solve(N,m,my_list,x0,Lbfgs_options)
        res2 = self.solve(N,x0,Lbfgs_options)
        if show_plot==True:
            import matplotlib.pyplot as plt
            t = np.linspace(0,self.T,N+1)
            u1 = res1['control'].array()[:N+1]
            u2 = res2['control'].array()
            plt.plot(t,u1)
            plt.plot(t,u2)
            plt.ylabel('control')
            plt.xlabel('time')
            plt.show()
            
        return res1,res2

    
    def lbfgs_memory_solve(self,N,m,my_list,mul=[1,2]):

        
        results = []
        for i in range(len(mul)):
            opt = {"mem_lim" : mul[i]*max(m,10)}
            try:
                res = self.penalty_solve(N,m,my_list,Lbfgs_options=opt)
            except Warning:
                res = {'iteration' : -1}
            results.append(res)
        return results

    def scipy_solver(self,N):

        dt=float(self.T)/N
        
            
        def J(u):
            return self.Functional(u,N)

        def grad_J(u):
            l = self.adjoint_solver(u,N)
            return self.grad_J(u,l,dt)

        res = minimize(J,np.zeros(N+1),method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': False})
        
        return res
        
        

        
class Problem1(OptimalControlProblem):
    """
    optimal control with ODE y=ay'+u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a)


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        return (1+dt*a)*l[-(i+1)] 

class Problem2(OptimalControlProblem):
    """
    optimal control with ODE y=a(t)y' + u
    """

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a


    def ODE_update(self,y,u,i,j,dt):
        a = self.a
        return (y[i] +dt*u[j+1])/(1.-dt*a(dt*(j+1)))


    def adjoint_update(self,l,y,i,dt):
        a = self.a
        
        return (1+dt*a(self.T-dt*i))*l[-(i+1)] 


class Problem3(OptimalControlProblem):
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
        return (1+dt*a)*l[-(i+1)] 

if __name__ == "__main__":

    from matplotlib.pyplot import *

    y0 = 1
    yT = 1
    T  = 1
    a  = 1

    def J(u,y,yT,T):
        t = np.linspace(0,T,len(u))

        I = trapz(u**2,t)

        return 0.5*(I + (y-yT)**2)

    def grad_J(u,p,dt):
        return dt*(u+p)


    problem = Problem1(y0,yT,T,a,J,grad_J)

    N = 1000
    m = 10
    
    res1 = problem.solve(N)

    t=np.linspace(0,T,N+1)
    u1 = res1['control'].array()[:N+1]
    figure()
    plot(t,u1)
    #show()


    opt = {"mem_lim":20}
    res2 = problem.penalty_solve(N,m,[500],Lbfgs_options=opt)

    u2 = res2['control'].array()[:N+1]

    plot(t,u2)
    show()
    

    
