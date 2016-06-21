from my_bfgs.lbfgs import Lbfgs
from penalty import partition_func
import numpy as np

class OptimalControlProblem():

    def __init__(self,y0,yT,T,J,grad_J,options=None):
        """
        Initialize OptimalControlProblem

        Valid options are:
        - T: End time 
        - J: Functional depending on u and y
        - options: More options
        """
        self.T      = T
        self.y0     = y0
        self.yT     = yT
        self.J      = J
        self.grad_J = grad_J
        
        self.set_options(options)

    def set_options(self,user_options):
        """
        Method for setting options
        """
        options = self.default_options()
        
        if user_options!=None:
            for key, val in user_options.iteritems():
                options[key]=val
        self.options = options

    def default_options(self):
        
        default = {"jtol"                   : 1e-4,
                   "maxiter"                : 200,
                   "mem_lim"                : 5,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : True,}

        return default

    
        
        
    def ODE_update(self,y,u,i,j,dt)
        raise NotImplementedError,'ODE_update not implemented'

    def adjoint_update(self,l,i,dt)
        raise NotImplementedError,'adjoint_update not implemented'
    
    def ODE_solver(self,u,N):
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        
        
        
        y = np.zeros(N+1)
        y[0]=y0

        for i in range(N):
            y[i+1] = self.ODE_update(y,u,i,i,dt)

    
        return y
    """
        else:

            
            y = partition_func(N+1,m)

            y[0][0]=y0
            for i in range(1,m):
            y[i][0] = u[N+i]

            start=1
            for i in range(m):        
                for j in range(len(y[i])-1):
                    y[i][j+1] = self.ODE_update(y[i],u,j,start+j)
                start = start + len(y[i]) - 1
            y_ret.append(y)

            Y = np.zeros(N+1)
            start = 0
            for i in range(m):
                Y[start:start+len(y[i])-1] = y[i][:-1]
                start = start + len(y[i]) - 1
            Y[-1] = y[-1][-1]
            
            y_ret.append(Y)
        return y_ret
    """
    def ODE_penalty_solver(self,u,N,m):
        
        T = self.T
        y0 = self.y0
        dt = float(T)/N
        y = partition_func(N+1,m)

        y[0][0]=y0
        for i in range(1,m):
            y[i][0] = u[N+i]

        start=1
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

        l=np.zeros(n+1)
        
        l[-1] = y[-1] -yT
        for i in range(N):
            l[-(i+2)] = self.adjoint_update(l,i,dt)
             
        return l
    """
        
        else:

            l = partition_func(N+1,m)
            y,Y = self.ODE_solver(u,N,m)

            
            l[-1][-1] = y[-1][-1] - yT
            for i in range(m-1):
                l[i][-1]=my*(y[i][-1]-u[N+1+i])

            for i in range(m):
                for j in range(len(l[i])-1):
                    l[i][-(j+2)] = self.adjoint_update(l[i],j,dt)
            l_ret.append(l)
            L=zeros(N+1)
            
            start=0
            for i in range(m):
                L[start:start+len(l[i])-1] = l[i][:-1]
                start = start + len(l[i])-1
            L[-1]=l[-1][-1]
            
            l_ret.append(L)
            
        return l_ret
    """
    
    def adjoint_penalty_solver(self,u,N,m,my):

        T = self.T
        y0 = self.y0
        dt = float(T)/N
        yT = self.yT
        
        l = partition_func(N+1,m)
        y,Y = self.ODE_penalty_solver(u,N,m)

            
        l[-1][-1] = y[-1][-1] - yT
        for i in range(m-1):
            l[i][-1]=my*(y[i][-1]-u[N+1+i])

        for i in range(m):
            for j in range(len(l[i])-1):
                l[i][-(j+2)] = self.adjoint_update(l[i],j,dt)
            
        L=zeros(N+1)
            
        start=0
        for i in range(m):
            L[start:start+len(l[i])-1] = l[i][:-1]
            start = start + len(l[i])-1
        L[-1]=l[-1][-1]
            
        return l,L
        
        
    def Functional(self,u,N):

        return self.J(u,self.ODE_solver(u,N)[-1],self.T)

    def Penalty_Functional(self,u,N,m,my):

        y,Y = self.ODE_penalty_solver(u,N,m)

        J_val = self.J(u[:N+1],y[-1][-1],self.T)

        penalty = 0

        for i in range(m-1):
            penalty = penalty + my*((y[i][-1]-u[N+i])**2)
    
        return J_val + 0.5*penalty


    
    
    def solve(self,N,x0=None,Lbfgs_options=None):

        
        dt=float(T)/n
        if x0==None:
            x0 = np.zeros(N+1)
            
        def J(u):
            return self.Functional(u,N)

        def grad_J(u):
            l = self.adjoint_solver(u,N)
            return self.grad_J(u,l,dt)

        if Lbfgs_options==None:
            Loptions = self.options
        else:
            Loptions = self.options
            for key, val in Lbfgs_options.iteritems():
                Loptions[key]=val
            
        
        solver = Lbfgs(x0,J,grad_J,options=Loptions)

        res = solver.solve()

        return res


    def penalty_solve(self,N,m,my_list,x0=None,Lbfgs_options=None):


        dt=float(T)/n
        if x0==None:
            x0 = np.zeros(N+m)
        x = None
        
        for i in range(len(my_list)):
        
            def J(u):                
                return Penalty_Functional(u,N,m,my_list[i])

            def grad_J(u):

                l,L = adjoint_penalty_solver(u,N,m,my_list[i])

                g = zeros(len(u))

                g[:N+1]=self.grad_J(u[:N+1],L,dt)

                for i in range(m-1):
                    g[N+1+i]=l[i+1][0]-l[i][-1]
                    
                return g
            
            if Lbfgs_options==None:
                Loptions = self.options
            else:
                Loptions = self.options
                for key, val in Lbfgs_options.iteritems():
                    Loptions[key]=val

            
            solver = Lbfgs(x0,J,grad_J,options=Loptions)
            
            res = solver.solve()
            
            x0 = res['control'].array()
            

        return res

class Problem1(OptimalControlProblem):

    def __init__(self,y0,yT,T,a,J,grad_J,options=None):

        OptimalControlProblem.__init__(self,y0,yT,T,J,grad_J,options)

        self.a = a
