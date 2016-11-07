import numpy as np
from linesearch.strong_wolfe import *


class OptimizationControl():

    def __init__(self,x0,J_f,grad_J,decomp=1):

        self.x = x0.copy()
        self.J_func = J_f
        self.grad_J = grad_J
        self.length = len(x0)

        self.dJ = grad_J(x0)

        self.niter = 0

    def update(self,x):
        
        self.x = x.copy()
        self.dJ = self.grad_J(x)
        self.niter += 1
        
    def v_update(self,N,v):
        self.x[:N+1] = v[:].copy()
        self.dJ = self.grad_J(x)
        
    def lamda_update(self,N,lamda):
        self.x[N+1:]=lamda[:]
        self.dJ = self.grad_J(x)
        self.niter += 1
        
    def val(self):
        return self.J_func(self.x)

class SteepestDecent():

    def __init__(self,J,grad_J,x0,decomp=1,options={}):

        self.J = J
        self.grad_J=grad_J
        self.x0 = x0
        self.decomp=decomp
        self.set_options(options)

        self.data = OptimizationControl(x0,J,grad_J)
        
        
        

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

    def do_linesearch(self,J,d_J,x,p):

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
        """
        if self.decomp!=1:
            N = len(y)-self.decomp
            y = y[:N+1]
        """
        if np.sqrt(np.sum(y**2)/len(y))<self.options['jtol']:
            print 'Success'
            return 1
            
        if k>self.options['maxiter']:
            return 1
        return 0


    def solve(self):
        import matplotlib.pyplot as plt
        J = self.J
        grad_J = self.grad_J
        opt = self.options
        k=1
        while self.check_convergence()==0:
            
            p = -self.data.dJ

            x,alfa = self.do_linesearch(J,grad_J,self.data.x,p)           
            self.data.update(x)
            plt.plot(x)
            
            print 'val: ',self.data.val()
        plt.show()
        return self.data


class PPCSteepestDecent(SteepestDecent):
    
    def __init__(self,J,grad_J,x0,PC,options={},decomp=1):
        SteepestDecent.__init__(self,J,grad_J,x0,
                                decomp=decomp,options=options)

    
        self.PC = PC

    def solve(self):

        J = self.J
        grad_J = self.grad_J
        opt = self.options
        import matplotlib.pyplot as plt

        while self.check_convergence()==0:

            p = -self.PC(self.data.dJ)

            x,alfa = self.do_linesearch(J,grad_J,self.data.x,p)           
            self.data.update(x)

            plt.plot(x[:len(x)-self.decomp+1])
            #plt.show()
            print 'val: ',self.data.val()
        plt.show()
        return self.data
    
    def split_control(self,x,m):
        N = len(x)-m
        X = x.copy()
        v = X[:N+1]
        lamda = X[N+1:]
        return v,lamda

    def get_v_grad(N,m,lamda):
        
        p_v = self.data.dJ[:N+1]

        def v_J(x_v):
            x = np.zeros(N+m)
            x[:N+1] = x_v[:]
            x[N+1] = lamda[:]
            return self.J(x)
            
        def v_grad(x_v):
            x = np.zeros(N+m)
            x[:N+1] = x_v[:]
            x[N+1] = lamda[:]

            return self.grad_J(x)[:N+1]
        
        return p_v,v_J,v_grad

    def get_lamda_grad(N,m,v):

        p_lamda = self.data.dJ[N+1:]

        def lamda_J(x_lamda):
            x = np.zeros(N+m)
            x[:N+1] = v[:]
            x[N+1] = x_lamda[:]
            return self.J(x)
            
        def lamda_grad(x_lamda):
            x = np.zeros(N+m)
            x[:N+1] = v[:]
            x[N+1] = x_lamda[:]

            return self.grad_J(x)[N+1:]

        return p_lamda,lamda_J,lamda_grad

    def split_solve(self,m):

        
        N = self.data.length
        opt = self.options

        while self.check_convergence()==0:

            v,lamda = self.split_control(self.data.x,m)

            p_v,v_J,v_grad = self.get_v_grad()
            v,alfa = self.do_linesearch(v_J,v_grad,v,p_v)
            self.data.v_update(v)
            
            p_lamda,lamda_J,lamda_grad = self.get_lamda_grad()
            
            lamda,alfa =  self.do_linesearch(lamda_J,lamda_grad,lamda,p_lamda)
            
            self.data.lamda_update(lamda)
        return self.data
        

if __name__ == "__main__":


    def J(x):

        return x.dot(x) + 1

    def grad_J(x):
        return 2*x

    x0 = np.zeros(30) + 1

    SD=SteepestDecent(J,grad_J,x0)

    res = SD.solve()

    print res.x
