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

        return self.data



    
    



    
