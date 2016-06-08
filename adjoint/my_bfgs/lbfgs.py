import numpy as np
from linesearch.strong_wolfe import *

from matplotlib.pyplot import *

from my_vector import SimpleVector, MuVector,MuVectors
from LmemoryHessian import LimMemoryHessian, MuLMIH


class LbfgsParent():

    def __init__(self,J,d_J,x0,Hinit=None,lam0=None,options=None):
        
        self.J    = J
        self.d_J  = d_J
        self.x0   = x0
        self.lam0 = lam0

        self.set_options(options)
        
        
        
        if Hinit==None:
            self.Hinit = np.identity(x0.size())
            


    def set_options(self,user_options):

        options = self.default_options()
        
        if user_options!=None:
            for key, val in user_options.iteritems():
                options[key]=val
        
            

        self.options = options
    
    def default_options(self):
        raise NotImplementedError, 'Lbfgs.default_options() not implemented' 
        

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
        raise NotImplementedError, 'Lbfgs.default_solve() not implemented'
########################################
##########################
############################
#########################     LBFGS
############################
##############################################

class Lbfgs(LbfgsParent):

    def __init__(self,J,d_J,x0,Hinit=None,lam0=None,options=None):

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=None,lam0=None,options=options)
        
        mem_lim = self.options['mem_lim']
        
        beta = self.options["beta"]
        
            
        Hessian = LimMemoryHessian(self.Hinit,mem_lim,beta=beta)
        self.data = {'control'   : x0,
                     'iteration' : 0,
                     'lbfgs'     : Hessian }

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
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : False,}
        
        return default

    
    def solve(self):
        
        x0=self.x0
        n=x0.size()
        x = SimpleVector(np.zeros(n))
        
        Hk = self.data['lbfgs']

        df0 = SimpleVector( self.d_J(x0.array()))
        df1 = SimpleVector(np.zeros(n))

        iter_k = self.data['iteration']
    
        p = SimpleVector(np.zeros(n))

        tol = self.options["jtol"]
        max_iter = self.options['maxiter']

        while np.sqrt(np.sum((df0.array())**2))/n>tol and iter_k<max_iter:

            #plot(np.linspace(0,1,n),x0.array())
            #show()
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
            self.data['iteration'] = iter_k
            self.data['control'] = x
            
        if self.options["return_data"] == True:
            return self.data

        return x

########################################
##########################
############################
#########################     MU LBFGS
############################
##############################################


class MuLbfgs(LbfgsParent):

    def __init__(self,J,d_J,x0,Mud_J,Hinit=None,lam0=None,options=None):

        LbfgsParent.__init__(self,J,d_J,x0,Hinit=None,lam0=None,options=options)

        self.Mud_J = Mud_J
        
        mem_lim  = self.options['mem_lim']
        beta     = self.options["beta"]
        mu       = self.options["mu_val"]
        H        = self.options["old_hessian"]
        save_num = self.options["save_number"]

        Hessian = MuLMIH(self.Hinit,mu=mu,H=H,mem_lim=mem_lim,beta=beta,
                         save_number=save_num)

        self.data = {'control'   : x0,
                     'iteration' : 0,
                     'lbfgs'     : Hessian }

    
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
                   "mem_lim"                : 5,
                   "Hinit"                  : "default",
                   "beta"                   : 1, 
                   "mu_val"                 : 1,
                   "old_hessian"            : None,
                   "penaly_number"          : 1,
                   "save_number"            :-1,          
                   "return_data"            : False, }
        
        return default


    def solve(self):
        
        x0=self.x0
        n=x0.size()
        m=self.options["penaly_number"]
        
        x = SimpleVector(np.zeros(n))
        
        Hk = self.data['lbfgs']
        
        

        mu = self.options["mu_val"]
        
        mu_df0, mu_x0 = self.Mud_J(x0)
        mu_df1 = None
        mu_x1  = None
        

        iter_k = self.data['iteration']


        df0 = SimpleVector(self.d_J(x0.array()))
        df1 = SimpleVector(np.zeros(n))

        p = SimpleVector(np.zeros(n))

        tol = self.options["jtol"]
        max_iter = self.options['maxiter']



        while np.sqrt(np.sum((df0.array())**2))/n>tol and iter_k<max_iter:

            
            
            p = Hk.matvec(-df0)
            
            
            x,alfa = self.do_linesearch(self.J,self.d_J,x0,p)

            df1.set(self.d_J(x.array()))           
            mu_df1,mu_x1 = self.Mud_J(x)


            Hk.update(mu_df1-mu_df0,mu_x1-mu_x0)



            mu_df0 = mu_df1.copy()
            mu_x0  = mu_x1.copy()
            
            
            
            x0=x.copy()
            df0=df1.copy()
            

            iter_k=iter_k+1
            self.data['iteration'] = iter_k
            self.data['control']   = x
        
        if self.options["return_data"]:
            return self.data

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
