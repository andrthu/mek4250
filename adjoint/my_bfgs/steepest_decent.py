import numpy as np
from linesearch.strong_wolfe import *


class OptimizationControl():

    def __init__(self,x0,J_f,grad_J):

        self.x = x0.copy()
        self.J_func = J
        self.grad_J = grad_J

        self.dJ = grad_J(x0)

        self.niter = 0

    def update(self,x):
        self.x = x.copy()
        self.dJ = self.grad_J(x)
        self.niter + = 1

    def val(self):
        return self.J_func(self.x)

class SteepestDecent():

    def __init__(self,J,grad_J,x0,options={}):

        self.J = J
        self.grad_J=grad_J
        self.x0 = x0

        self.set_options(options)

        self.data = OptimizationControl(x0,J_f,grad_J)

        

    def set_options(self,user_options):

        options = self.default_options()

        for key, val in user_options.iteritems():
            options[key]=val

        self.options = options

    def default_options(self):
        default = {}
        ls = {"ftol": 1e-3, "gtol": 0.9, "xtol": 1e-1, "start_stp": 1}
        default.update(            
            {"jtol"                   : 1e-4,
             "gtol"                   : 1e-4,
             "maxiter"                :  200,
             "line_search_options"    : ls,})
        return default

    def do_linesearch(self,J,d_J,x0,p):

        x_new = x.copy()
    
        

        def phi(alpha):
            """
            Convert functional to a one variable functon dependent
            on step size alpha
            """
            
            x_new=x.copy()
            x_new = x_new + alpha*p.copy()
            return J(x_new)
    
        def phi_dphi(alpha):
            """
            Derivative of above function
            """
            x_new = x.copy()
        
            x_new = x_new +alpha*p.copy()
        
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
        x_new=x_new + alpha*p.copy()
    
        return x_new, float(alpha)


    def check_convergence(self):
        
        y = self.data.dJ
        k = self.data.niter

        if np.sqrt(np.sum(y**2)/len(y))<self.options['jtol']:
            return 1
            
        if k>self.options['maxiter']:
            return 1
        return 0


    def solve(self):

        J = self.J
        grad_J = self.grad_J
        opt = self.options
        
        while self.check_convergence()==0:

            p = -self.data.dJ

            x,alfa = self.do_linesearch(J,grad_J,self.data.x,p)           
            self.data.update(x)

    return self.data
