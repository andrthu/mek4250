import numpy as np
from lbfgs import LbfgsParent
from linesearch.strong_wolfe import *

from my_vector import SimpleVector
from LmemoryHessian import LimMemoryHessian,NumpyLimMemoryHessian
from lbfgsOptimizationControl import LbfgsOptimizationControl

class SplitLbfgs(LbfgsParent):

    def __init__(self,J,d_J,x0,m,Hinit=None,options=None):

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=Hinit,options=options)

        self.m = m

        mem_lim = self.options['mem_lim']
        
        beta = self.options["beta"]
        
            
        Hessian = NumpyLimMemoryHessian(self.Hinit,mem_lim,beta=beta)

        self.data = LbfgsOptimizationControl(x0,J,d_J,Hessian)

    def do_linesearch(self,J,d_J,x,p):
        """
        Method that does a linesearch using the strong Wolfie condition

        Arguments:

        * J : The functional
        * d_J : The gradient
        * x : The starting point of the linesearch
        * p : The direction of the search, i.e. -d_J(x)

        Return value:
        
        * x_new : The ending point of the linesearch
        * alpha : The step length
        """
        x_new = x.copy()
    
        

        def phi(alpha):
            """
            Convert functional to a one variable functon dependent
            on step size alpha
            """
            
            x_new=x.copy()
            x_new=x_new +alpha*p
            return J(x_new)
    
        def phi_dphi(alpha):
            """
            Derivative of above function
            """
            x_new = x.copy()
        
            x_new = x_new + alpha*p
        
            f = J(x_new)
            djs = p.dot(d_J(x_new))
        
            return f,float(djs)
            
        
        phi_dphi0 = J(x),float(p.dot(d_J(x)))
        
        
        if self.options["line_search"]=="strong_wolfe":
            ls_parm = self.options["line_search_options"]
            
            ftol      = ls_parm["ftol"]
            gtol      = ls_parm["gtol"]
            xtol      = ls_parm["xtol"]
            start_stp = ls_parm["start_stp"]
            
            ls = StrongWolfeLineSearch(ftol,gtol,xtol,start_stp,
                                        ignore_warnings=False)

        alpha = ls.search(phi, phi_dphi, phi_dphi0)

        
        x_new=x.copy()
        x_new=x_new+alpha*p
    
        return x_new, float(alpha)



    def default_options(self):

        ls = {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1}
        
        default = {"jtol"                   : 1e-4,
                   "gtol"                   : 1e-4,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,                   
                   "mem_lim"                : 10,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : False,}
        
        return default



    def check_convergence(self):
        
        y = self.data.dJ
        k = self.data.niter
        
        if np.sqrt(np.sum(y**2)/len(y))<self.options['jtol']:
            print 'Success'
            return 1
            
        if k>self.options['maxiter']:
            return 1
        return 0
    

    def solve(self):
        
        n = self.data.length
        m = self.m
        N = n-m
        x0 = self.data.x.copy()

        H = self.data.H
        df0 = self.data.dJ.copy()
        df1 = np.zeros(n)

        while self.check_convergence()==0:
            
            p = H.matvec(-df0)
            #"""
            v,lamda =self.split_control(self.data.x,N,m)

            p_lamda,lamda_J,lamda_grad = self.get_lamda_grad(p,N,m,v,lamda)
            
            lamda1,alpha = self.do_linesearch(lamda_J,lamda_grad,lamda,p_lamda)

            p_v,v_J,v_grad = self.get_v_grad(p,N,m,v,lamda)
            v,alpha = self.do_linesearch(v_J,v_grad,v,p_v)

            
            
            
            self.data.split_update(N,v,lamda1)
            #"""
            #x,alpha = self.do_linesearch(self.J,self.d_J,x0,p)
            #self.data.update(x)
            df1[:]=self.data.dJ[:].copy()

            s = self.data.x-x0
            y = df1-df0

            H.update(y,s)
            x0[:]=self.data.x[:]
            df0[:]=df1[:]
            
        
        return self.data

    def split_control(self,x,N,m):
        #N = len(x)-m
        X = x.copy()
        v = X[:N+1]
        lamda = X[N+1:]
        return v,lamda

    def get_v_grad(self,p,N,m,v,lamda):
        
        p_v = p[:N+1]

        def v_J(x_v):
            x = np.zeros(N+m)
            
            x[:N+1] = x_v[:]
            x[N+1:] = lamda[:]
            return self.J(x)
            
        def v_grad(x_v):
            x = np.zeros(N+m)
            x[:N+1] = x_v[:]
            x[N+1:] = lamda[:]

            return self.d_J(x)[:N+1]
        
        return p_v,v_J,v_grad

    def get_lamda_grad(self,p,N,m,v,lamda):

        p_lamda = p[N+1:]

        def lamda_J(x_lamda):
            x = np.zeros(N+m)
            x[:N+1] = v[:]
            x[N+1:] = x_lamda[:]
            return self.J(x)
            
        def lamda_grad(x_lamda):
            x = np.zeros(N+m)
            x[:N+1] = v[:]
            x[N+1:] = x_lamda[:]

            return self.d_J(x)[N+1:]
        #p_lamda = -self.PC(lamda_grad(lamda))

        return p_lamda,lamda_J,lamda_grad

    
    



    
