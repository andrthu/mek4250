import numpy as np
from lbfgs import LbfgsParent

from my_vector import SimpleVector
from LmemoryHessian import LimMemoryHessian,NumpyLimMemoryHessian
from lbfgsOptimizationControl import LbfgsOptimizationControl

class SplitLbfgs(LbfgsParent):

    def __init__(self,J,d_J,x0,m,Hinit=None,options=None):

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=Hinit,options=options)

        self.m = m

        mem_lim = self.options['mem_lim']
        
        beta = self.options["beta"]
        
            
        Hessian = LimMemoryHessian(self.Hinit,mem_lim,beta=beta)

        self.data = LbfgsOptimizationControl(x0,J,d_J,Hessian)



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
        x0 = self.data.x

        H = self.data.H
        df0 = self.data.dJ
        df1 = np.zeros(n)
        while self.check_convergance()==0:
            
            p = H.matvec(-df0)
            v,lamda =self.split_control(self.x,N,m)

            p_v,v_J,v_grad = self.get_v_grad(p,N,m,v,lamda)
            v,alpha = self.linesearch(v_J,v_grad,v,p_v)

            p_lamda,lamda_J,lamda_grad = self.get_lamda_grad(p,N,m,v,lamda)
            
            lamda,alpha = self.linesearch(lamda_J,lamda_grad,lamda,p_lamda)
            
            
            self.data.update(N,v,lamda)
            df1[:]=self.data.dJ[:]

            s = self.x-x0
            y = df1-df0

            H.update(y,s)
            x0[:]=self.x[:]
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

            return self.grad_J(x)[:N+1]
        
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

            return self.grad_J(x)[N+1:]
        #p_lamda = -self.PC(lamda_grad(lamda))

        return p_lamda,lamda_J,lamda_grad

    
    



    
