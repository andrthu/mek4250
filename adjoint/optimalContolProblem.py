from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np

class OptimalControlProblem():
    """
    class for solving problem on the form

    minimize J(y(u),u) = A(u) + B(y(T),yT)
    E(y(t),u(t)) = 0
    """

    def __init__(self,y0,yT,T,J,grad_J,Lbfgs_options=None,options=None):
        """
        Initialize OptimalControlProblem

        Valid options are:
        * y0: initial condition of the ODE
        * yT: 'End goal' of the ODE in the functional
        * T: End time 
        * J: Functional depending on u,y,T and yT.
        * grad_J : gradient for above Functional, depends on u, p and dt 
        * Lbfgs_options: options for the L-bfgs algorithm for solving
          - jtol: Tolorance for end criteria in L-BFGS
          - maxiter: maximal number of iterations
          - mem_lim: number of iterations remembered by inverted hessian
          - Vector: Vector type
          - Hinit: initial approximation to inverted hessian
          - beta: scalar multiplied with Hinit
          - return_data: output is just control, or more information
        * options: More options
          - Vector: What type of vector shall be used. 
          - Lbfgs: which L-bfgs implementation shall be used
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
        """
        default options for LGFGS
        """
        default = {"jtol"                   : 1e-6,
                   "maxiter"                : 500,
                   "mem_lim"                : 10,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : True,}

        return default

    def default_options(self):
        """
        Default options
        """

        default = {"Vector" : SimpleVector,
                   "Lbfgs"  : Lbfgs,}
        return default
        
        
    def ODE_update(self,y,u,i,j,dt):
        """
        Finite diffrence scheme for state equation
        """
        raise NotImplementedError,'ODE_update not implemented'

    def adjoint_update(self,l,y,i,dt):
        """
        Finite diffrence scheme for adjoint equation
        """
        raise NotImplementedError,'adjoint_update not implemented'

    def initial_adjoint(self,y):
        """
        Initial condition for adjoint equation. Depends on the Functional
        """
        return y - self.yT

    def initial_penalty(self,y,u,my,N,i):
        """
        initial conditian for the adjoint equations, when partitioning in time
        """
        return my*(y[i][-1]-u[N+1+i])

    def initial_lagrange(self,y,u,my,N,i,G):
        """
        Same as above, when we use augmented Lagrange
        """
        return self.initial_penalty(y,u,my,N,i) + G[i]
    
    def ODE_solver(self,u,N):
        """
        Solving the state equation without partitoning

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        
        
        
        y = np.zeros(N+1)
        y[0]=y0

        for i in range(N):
            y[i+1] = self.ODE_update(y,u,i,i,dt)

    
        return y

    def ODE_penalty_solver(self,u,N,m):
        """
        Solving the state equation with partitioning

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """
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
        """
        Solving the adjoint equation using finite difference

        Arguments:
        * u: the control
        * N: Number of discritization points
        """
        
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
    
    def adjoint_penalty_solver(self,u,N,m,my,init=initial_penalty):
        """
        Solving the adjoint equation using finite difference, and partitioning
        the time interval.

        Arguments:
        * u: the control
        * N: Number of discritization points
        * m: Number of intervals we partition time in
        """

        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        
        l = partition_func(N+1,m)
        y,Y = self.ODE_penalty_solver(u,N,m)

            
        l[-1][-1] = self.initial_adjoint(y[-1][-1])
        for i in range(m-1):
            l[i][-1]=init(self,y,u,my,N,i) #my*(y[i][-1]-u[N+1+i])

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

    def Lagrange_Penalty_Functional(self,u,N,m,my,G):
        """
        Add penalty terms and lagrangian terms to the functional, for
        augmented lagrange method

        Arguments:

        * u: control
        * N: discretization points
        * m: number of parttition intervalls
        * my: penalty variable
        * G: lagrange multiplier
        
        """
        y,Y = self.ODE_penalty_solver(u,N,m)

        J_val = self.J(u[:N+1],y[-1][-1],self.yT,self.T)

        pen = 0

        for i in range(m-1):
            pen = pen + (my*(y[i][-1]-u[N+1+i])+2*G[i])*(y[i][-1]-u[N+1+i])
    
        return J_val + 0.5*pen

    
        
    def Functional(self,u,N):
        """
        Reduced functional, that only depend on control u 
        """

        return self.J(u,self.ODE_solver(u,N)[-1],self.yT,self.T)

    def Penalty_Functional(self,u,N,m,my):
        """
        Reduced functional, that only depend on control u. Also adds
        penalty terms
        """
        y,Y = self.ODE_penalty_solver(u,N,m)

        J_val = self.J(u[:N+1],y[-1][-1],self.yT,self.T)

        penalty = 0

        for i in range(m-1):
            penalty = penalty + my*((y[i][-1]-u[N+1+i])**2)
    
        return J_val + 0.5*penalty


    
    
    def solve(self,N,x0=None,Lbfgs_options=None):
        """
        Solve the optimazation problem without penalty

        Arguments:
        * N: number of discritization points
        * x0: initial guess for control
        * Lbfgs_options: same as for class initialisation
        """
        
        dt=float(self.T)/N
        if x0==None:
            x0 = self.Vec(np.zeros(N+1))
        else:
            x0 = self.Vec(x0)
            
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
        """
        Solve the optimazation problem with penalty

        Arguments:
        * N: number of discritization points
        * m: number ot processes
        * my_list: list of penalty variables, that we want to solve the problem
                   for.
        * x0: initial guess for control
        * Lbfgs_options: same as for class initialisation
        """

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
                    g[N+1+j]= l[j+1][0] - l[j][-1]
                    
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
            x0 = res['control']
            
        if len(Result)==1:
            return res
        else:
            return Result

    def lagrange_penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None):
        """
        Solve the optimazation problem with augmented lagrange

        Arguments:
        * N: number of discritization points
        * m: number ot processes
        * my_list: list of penalty variables, that we want to solve the problem
                   for.
        * x0: initial guess for control
        * Lbfgs_options: same as for class initialisation
        """

        dt=float(self.T)/N
        if x0==None:
            x0 = self.Vec(np.zeros(N+m))
        x = None
        Result = []

        G = np.zeros(m-1)
        for i in range(len(my_list)):
            print
            print my_list[i],G
            print 
            def init_pen(self,y,u,my,N,k):
                return self.initial_lagrange(y,u,my,N,k,G)
            
            def J(u):                
                return self.Lagrange_Penalty_Functional(u,N,m,my_list[i],G)

            def grad_J(u):

                l,L = self.adjoint_penalty_solver(u,N,m,my_list[i],init=init_pen )

                g = np.zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)

                for j in range(m-1):
                    g[N+1+j]=l[j+1][0]-l[j][-1] - G[j]
                    
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
            x0 = res['control']
            print 
            y,Y = self.ODE_penalty_solver(res['control'].array(),N,m)
            for j in range(m-1):
                G[j]=G[j]-my_list[i]*(y[j][-1]-res['control'].array()[N+1+j])

        if len(Result)==1:
            return res
        else:
            return Result


        
    def plot_solve(self,N,x0=None,opt=None,state=False):
        """
        Solve the problem and plot the control

        Arguments:

        * N: # discrite points
        * x0: inital gues
        * opt: optians for Lbfgs
        * state: if not false, method also plots the state
        """
        res = self.solve(N,x0=x0,Lbfgs_options=opt)
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
        """
        solves problem using both normal and penalty approach 
        """
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
        """
        Method used in testing
        """
        
        results = []
        for i in range(len(mul)):
            opt = {"mem_lim" : mul[i]*max(m,10)}
            try:
                res = self.penalty_solve(N,m,my_list,Lbfgs_options=opt)
            except Warning:
                res = {'iteration' : -1}
            results.append(res)
        return results

    def simple_test(self,N,make_plot=False):

        if make_plot:
            teller=0
            import matplotlib.pyplot as plt
            t = np.linspace(0,self.T,N+1)

        M = [2,4,8,16,32]
        L=[]
        mul=[1,3]
        for i in range(len(M)):

            L.append(self.lbfgs_memory_solve(N,M[i],[0.5*N],mul=mul))
        try:
            res1 = self.solve(N)
        except Warning:
            res1 = {'iteration' : -1}


        print "--------------m=1--------------" 
        print "|lbfgs memory=10| #iterations=%d| #iterations/m=%.2f"%(res1['iteration'],res1['iteration']) 
        if make_plot:
            if res1['iteration']!=-1:
                plt.plot(t,res1['control'].array(),'r--')
        for i in range(len(M)):
            print "--------------m=%d--------------" %(M[i])
            for j in range(len(mul)):
                print "|lbfgs memory=%d| #iterations=%d| #iterations/m=%.2f"%(mul[j]*max(M[i],10),L[i][j]['iteration'],L[i][j]['iteration']/float(M[i]))

                if make_plot:
                    
                    if j == len(mul)-1:
                        if L[i][j]['iteration']!=-1:
                            plt.plot(t,L[i][j]['control'].array()[:N+1])
                            teller =teller+1
        if make_plot:
            
            if teller==len(M):
                plt.legend(['m=1','m=2','m=4','m=8','m=16','m=32'],loc=4)
            plt.title('Controlls for non linear in y term problem')
            plt.xlabel('time')
            plt.ylabel('controls')
            plt.show()
    def scipy_solver(self,N,disp=False):
        """
        solve the problem using scipy LBFGS instead of self made LBFGS
        """
        dt=float(self.T)/N
        
            
        def J(u):
            return self.Functional(u,N)

        def grad_J(u):
            l = self.adjoint_solver(u,N)
            return self.grad_J(u,l,dt)

        res = minimize(J,np.zeros(N+1),method='L-BFGS-B', jac=grad_J,
                       options={'gtol': 1e-6, 'disp': disp})
        
        return res

    def scipy_penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None):
        
        dt=float(self.T)/N
        if x0==None:

            x0 = np.zeros(N+m)

        Result = []

        for i in range(len(my_list)):
        
            def J(u):                
                return self.Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):

                l,L = self.adjoint_penalty_solver(u,N,m,my_list[i])

                g = np.zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)

                for j in range(m-1):
                    g[N+1+j]= l[j+1][0] - l[j][-1]
                    
                return g

            res = minimize(J,x0,method='L-BFGS-B', jac=grad_J,
                           options={'gtol': 1e-6, 'disp': False,
                                    'maxcor':20})

            Result.append(res.copy())

        if len(Result)==1:
            return res
        else:
            return Result

    def finite_diff(self,u,N):
        
        import matplotlib.pyplot as plt
        eps = 1./10000

        finite_grad_J = np.zeros(len(u))

        for i in range(len(u)):
            e = np.zeros(len(u))
            e[i]=eps
            J1 = self.Functional(u,N)
            J2 = self.Functional(u+e,N)
            finite_grad_J[i] = (J2-J1)/eps

        l = self.adjoint_solver(u,N)
        
        grad_J = self.grad_J(u,l,self.T/float(N))
        t = np.linspace(0,self.T,len(u))
        plt.plot(t,grad_J)
        plt.plot(t,finite_grad_J,'r--')
        plt.legend(['adjoint','finite diff'])
        plt.show()

        
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
    

    
