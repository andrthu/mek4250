import numpy as np
from linesearch.strong_wolfe import *

from my_vector import SimpleVector
from LmemoryHessian import LimMemoryHessian


class Lbfgs():

    def __init__(self,J,d_J,x0,Hinit=None,lam0=None,options=None):
        
        self.J    = J
        self.d_J  = d_J
        self.x0   = x0
        self.lam0 = lam0

        self.set_options(options)

        mem_lim = self.options['mem_lim']
        
        if Hinit==None:
            self.Hinit = np.identity(x0.size())

        self.data = {'control'   : x0,
                     'iteration' : 0,
                     'lbfgs'     : LimMemoryHessian(self.Hinit,mem_lim) }

    def set_options(self,user_options):

        if user_options!=None:
            options=user_options  
        else:
            options = self.default_options()

        self.options = options
    
    def default_options(self):
        ls = {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1}
        
        default = {"jtol"                   : 1e-4,
                   "rjtol"                  : 1e-6,
                   "gtol"                   : 1e-4,
                   "rgtol"                  : 1e-5,
                   "maxiter"                :  200,
                   "display"                :    2,
                   "line_search"            : "strong_wolfe",
                   "line_search_options"    : ls,
                   # method specific parameters:
                   "mem_lim"                : 5,
                   "Hinit"                  : "default",}
        
        return default


    def do_linesearch(self,J,d_J,x,p):
        
        x_new = x.copy()
    
    

        def phi(alpha):
            
            x_new=x.copy()
            x_new.axpy(alpha,p)
            return J(x_new.array())
    
        def phi_dphi(alpha):
        
            x_new = x.copy()
        
            x_new.axpy(alpha,p)
        
            f = J(x_new.array())
            djs = p.dot(SimpleVector(d_J(x_new.array())))
        
            return f,float(djs)
            
        #print p.dot(SimpleVector(d_J(x.array())))
        phi_dphi0 = J(x.array()),float(p.dot(SimpleVector(d_J(x.array()))))
        #print phi_dphi0
        
        if self.options["line_search"]=="strong_wolfe":
            ls_parm = self.options["line_search_options"]
            
            ftol     = ls_parm["ftol"]
            gtol     = ls_parm["gtol"]
            xtol     = ls_parm["xtol"]
            start_stp = ls_parm["start_stp"]
            
            ls =  StrongWolfeLineSearch(ftol,gtol,xtol,start_stp)

        alpha = ls.search(phi, phi_dphi, phi_dphi0)

        
        x_new=x.copy()
        x_new.axpy(alpha,p)
    
        return x_new, float(alpha)

    def solve(self):

        x0=self.x0
        n=x0.size()
        x = SimpleVector(np.zeros(n))
        
        Hk = self.data['lbfgs']

        df0 = SimpleVector( self.d_J(x0.array()))
        df1 = SimpleVector(np.zeros(n))

        iter_k = 0
    
        p = SimpleVector(np.zeros(n))

        tol = self.options["jtol"]
        max_iter = self.options['maxiter']

        while np.sqrt(np.sum((df0.array())**2))/n>tol and iter_k<max_iter:

        
            p = Hk.matvec(-df0)
            #print df0.array()
            #print p.array()
            x,alfa = self.do_linesearch(self.J,self.d_J,x0,p)

            df1.set(self.d_J(x.array()))
            
            s = x-x0
            y = df1-df0

            Hk.update(y,s)
             
            x0=x.copy()
            df0=df1.copy()

            iter_k=iter_k+1

        return x

if __name__== "__main__":

    
    def J(x):

        s=0
        for i in range(len(x)):
            s = s + (x[i]-1)**2
        return s


    def d_J(x):

        return 2*(x-1)

    x0=SimpleVector(np.linspace(1,30,30))
    

    solver = Lbfgs(J,d_J,x0)
    
    print solver.solve().array()
