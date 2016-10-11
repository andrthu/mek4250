from dolfin import *
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from my_bfgs.steepest_decent import SteepestDecent
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize as Mini
import numpy as np
from my_bfgs.my_vector import SimpleVector
import time

def time_partition(start,end,Tn,m):

    N = np.zeros(m)
    T = np.zeros(m+1) 

    timestep = float(end-start)/Tn

    fraq = Tn/m
    rest = Tn%m

    t = start
    T[0] = t

    for i in range(0,m):
        if rest-i >0:
            N[i] = fraq +1
        else:
            N[i] = fraq 

        t = t + timestep*N[i]
        T[i+1] = t

    return N,T
def partition_start(Tn,m):
    
    
    fraq = Tn/m
    rest = Tn%m
    
    start = []
    start.append(0)
    
    for i in range(m-1):
        if rest - i >0:
            start.append(start[i] + fraq + 1)
        else:
            start.append( start[i] + fraq) 
    return start
    
    
class FenicsOptimalControlProblem():

    def __init__(self,V,mesh,options={}):

        self.V = V
        self.mesh = mesh
        
        
        self.Lbfgs_options = self.default_Lbfgs_options()
        self.SD_options = self.default_SD_options()

    def default_Lbfgs_options(self):
        """
        default options for LGFGS
        """
        default = {"jtol"                   : 1e-4,
                   "maxiter"                : 500,
                   "mem_lim"                : 10,
                   "Vector"                 : SimpleVector,
                   "Hinit"                  : "default",
                   "beta"                   : 1,
                   "return_data"            : True,}

        return default

    def dfault_SD_options(self):
        default = {}
        default.update({"jtol"                   : 1e-4,
                        "maxiter"                :  200,})
        return default
    

    def adjoint_ic(self,opt,U):
        raise NotImplementedError, 'adjoint_ic not implemented'

    def PDE_form(self,ic,opt,u,u_,v,rhs,timestep):
        raise NotImplementedError, 'PDE_form not implemented'

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        raise NotImplementedError, 'adjoint_form not implemented'

    def get_control(self,opt,ic,m):
        raise NotImplementedError, 'get_control not implemented'

    def get_opt(self,control,opt,ic,m):
        raise NotImplementedError, 'get_opt not implemented'

    def grad_J(self,P,opt,ic,h):
        raise NotImplementedError, 'grad_J not implemented'
    
    def J(self,opt,ic,U,start,end):
        raise NotImplementedError, 'J not implemented'

    def penalty_grad_J(self,P,opt,ic,m,h):
        raise NotImplementedError, 'penalty_grad_J not implemented'

    def rhs_finder(self,Rhs,rhs,i):
        return None

    def penalty_J(self,opt,ic,U,start,end,Tn,m,mu):

        
        N,T=time_partition(start,end,Tn,m)

        s = 0
        for i in range(m):
            s += self.J(opt,U[i][0],U[i],T[i],T[i+1])
        penalty = 0
        for i in range(len(U)-1):
            penalty = penalty +0.5*assemble((U[i][-1]-U[i+1][0])**2*dx)
    
        return s + mu*penalty

                
        
    def PDE_solver(self,ic,opt,start,end,Tn,show_plot=False,first_step=0):

        U = []

        u_ = ic.copy()
        if show_plot:
            plot(u_)
            time.sleep(0.1)
        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)
        
        timestep = Constant((end-start)/float(Tn))
        Rhs = opt['rhs'] 
        rhs = Function(self.V)
        
        t_step = first_step
        self.rhs_finder(Rhs,rhs,t_step)

        F,bc = self.PDE_form(ic,opt,u,u_,v,rhs,timestep)


        t  = start

        while (t<end - DOLFIN_EPS):
            t_step+=1
            self.rhs_finder(Rhs,rhs,t_step)
            
            solve(F==0,u,bc)
            u_.assign(u)
            U.append(u_.copy())

            t += float(timestep)
            
            if show_plot:
                plot(u_)
                time.sleep(0.11)
        return U

    def adjoint_interval_solver(self,opt,p_ic,U,start,end,Tn):
        
        P = []

        timestep = Constant((end-start)/float(Tn))

        
        p_ = p_ic.copy()
        P.append(p_.copy())

        p = Function(self.V)
        v = TestFunction(self.V)
        u = U[-1]
        
        F,bc = self.adjoint_form(opt,u,p,p_,v,timestep)

        t = end
        count = -2
        while ( t> start + DOLFIN_EPS):

            solve(F==0,p,bc)
            p_.assign(p)
            u.assign(U[count].copy())
        
            P.append(p_.copy())
            count = count -1
            t = t - float(timestep)
    
        return P

        
    def adjoint_solver(self,ic,opt,start,end,Tn):

        U = self.PDE_solver(ic,opt,start,end,Tn)
        p_ic = self.adjoint_ic(opt,U)
        
        return self.adjoint_interval_solver(opt,p_ic,U,start,end,Tn)


    def penalty_PDE_solver(self,opt,ic,lam_ic,start,end,Tn,m):

        N,T = time_partition(start,end,Tn,m)
        s = partition_start(Tn,m)

        U = []
        V = self.V
        #PDE_solver(self,ic,opt,start,end,Tn,show_plot=False)
        U.append(self.PDE_solver(ic,opt,T[0],T[1],N[0],first_step=0))
        for i in range(1,m):
            U.append(self.PDE_solver(lam_ic[i-1],opt,T[i],T[i+1],N[i],
                                     first_step=s[i]))

        return U

    def penalty_adjoint_solver(self,ic,lam_ic,opt,start,end,Tn,m,mu):

        N,T = time_partition(start,end,Tn,m)
    
        U =  self.penalty_PDE_solver(opt,ic,lam_ic,start,end,Tn,m)
        
        P = []
    
        for i in range(m-1):
            P_ic = project((U[i][-1]-lam_ic[i])*Constant(mu),self.V)
            sta = T[i]
            en = T[i+1]
            P.append(self.adjoint_interval_solver(opt,P_ic,U[i],sta,en,N[i]))

        P_ic = self.adjoint_ic(opt,U[-1])
        #adjoint_interval_solver(self,opt,p_ic,U,start,end,Tn):
        P.append(self.adjoint_interval_solver(opt,P_ic,U[-1],T[-2],T[-1],N[-1]))

        return P

    def solver(self,opt,ic,start,end,Tn,algorithm='scipy_lbfgs',
               options=None):
        h = self.mesh.hmax()
        
        def J(x):
            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            U = self.PDE_solver(loc_ic,loc_opt,start,end,Tn)
            return self.J(loc_opt,loc_ic,U,start,end)
        
        def grad_J(x):

            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            P = self.adjoint_solver(loc_ic,loc_opt,start,end,Tn)

            return self.grad_J(P,loc_opt,loc_ic,h)


        control0 = self.get_control(opt,ic,1)
        if algorithm=='my_lbfgs':
            control0 = SimpleVector(control0)

            if options==None:
                Loptions = self.Lbfgs_options
            else:
                Loptions = self.Lbfgs_options
                for key, val in options.iteritems():
                    Loptions[key]=val


            solver = Lbfgs(J,grad_J,control0,options=Loptions)
        
            res = solver.solve()
        elif algorithm==scipy_lbfgs:
            res = Mini(J,control0.copy(),method='L-BFGS-B', 
                       jac=grad_J,options={'gtol': 1e-6, 'disp': True})

        elif algorithm=='my_steepest_decent':

            if options==None:
                opt = self.SD_options
            else:
                opt = self.SD_options
                for key, val in options.iteritems():
                    opt[key]=val



            Solver = SteepestDecent(J,grad_J,control0.copy(),opt)
            res = Solver.solve()
        return res


    def penalty_solver(self,opt,ic,start,end,Tn,m,mu_list,
                       algorithm='scipy_lbfgs',options=None):

        h = self.mesh.hmax()
        X = Function(self.V)
        xN = len(ic.vector().array())
        control0 = self.get_control(opt,ic,m)
        if algorithm=='my_lbfgs':
            control0 = SimpleVector(control0)
        
        res =[]
        for k in range(len(mu_list)):
            def J(x):
            
                cont_e = len(x)-(m-1)*xN 
                loc_opt,loc_ic = self.get_opt(x[:cont_e],opt,ic,m)

                lam = []
                
                for i in range(m-1):
                    l = Function(self.V)
                    l.vector()[:] = x.copy()[cont_e+i*xN:cont_e+(i+1)*xN]
                    lam.append(l.copy())
            
            

                U = self.penalty_PDE_solver(loc_opt,loc_ic,lam,start,end,Tn,m)
                mu = mu_list[k]
                return self.penalty_J(loc_opt,loc_ic,U,start,end,Tn,m,mu)
        
            def grad_J(x):

                cont_e = len(x)-(m-1)*xN 
                lopt,lic = self.get_opt(x[:cont_e],opt,ic,m)
            
                lam = []
                
                for i in range(m-1):
                    l = Function(self.V)
                    l.vector()[:] = x.copy()[cont_e+i*xN:cont_e+(i+1)*xN]
                    lam.append(l.copy())
                mu = mu_list[k]
                P=self.penalty_adjoint_solver(lic,lam,lopt,start,end,Tn,m,mu)

                return self.penalty_grad_J(P,lopt,lic,m,h)


            
            if algorithm == 'my_steepest_decent':
                if options==None:
                    opt = self.SD_options
                else:
                    opt = self.SD_options
                    for key, val in options.iteritems():
                        opt[key]=val



                Solver = SteepestDecent(J,grad_J,control0.copy(),opt)
                res1 = Solver.solve()

                control0 = res1.x.copy()
            else:
                
                if options==None:
                    Loptions = self.Lbfgs_options
                else:
                    Loptions = self.Lbfgs_options
                    for key, val in options.iteritems():
                        Loptions[key]=val

                if algorithm=='my_lbfgs':
                    solver = Lbfgs(J,grad_J,control0,options=Loptions)

                    res1 = solver.solve()
                    control0 = res1['control'].copy()
                elif algorithm=='scipy_lbfgs':
                    res1 = Mini(J,control0.copy(),method='L-BFGS-B',jac=grad_J,
                                options={'gtol':1e-6, 'disp':True,'maxcor':10})
                    control0 = res1.x.copy()


            res.append(res1)
        if len(res)==1:
            
            return res[0]
        
        return res


    def gather_penalty_funcs(self,L):
        l = []
        m = len(L)
        for i in range(len(L[0])):
            l.append(L[0][i])

        for i in range(1,m):
            for j in range(1,len(L[i])):
                l.append(L[i][j])

        return l



    

class Burger1(FenicsOptimalControlProblem):
    """
    u_t  +uu_x -u_xx =0
    """
    #opt = {nu : ...}
    def adjoint_ic(self,opt,U):
        """
        Initial condition of the adjoint
        """
        return project(Constant(0.0),self.V)

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

    def PDE_form(self,ic,opt,u,u_,v,rhs,timestep):
        """
        The PDE written on wariational form

        return
        -F: the form
        -bc: the boundery conditions
        """
        nu = Constant(opt['nu'])
        F = (self.Dt(u,u_,timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")
        return F,bc

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        """
        The adjoint PDE written on wariational form

        return
        -F: the form
        -bc: the boundery conditions
        """
        nu = Constant(opt['nu'])

        F = -(-self.Dt(p,p_,timestep)*v + u*p.dx(0)*v -nu*p.dx(0)*v.dx(0) +2*u*v)*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")
        return F,bc

    def get_control(self,opt,ic,m):
        """
        Function that returns the control
        """
        if m==1:
            return ic.copy().vector().array()
        
        N = len(ic.vector().array())
        x = np.zeros(m*N)
        x[:N] = ic.copy().vector().array()[:]

        return x
        
    def get_opt(self,control,opt,ic,m):
        """
        Given control overall options and ic, returns
        options and initial conitions, that might depend
        on control and ic.
        """
        g = Function(self.V)

        g.vector()[:] = control[:]

        return opt,g

    def grad_J(self,P,opt,ic,h):
        """
        what the gradient with respect to control looks like
        """
        return h*P[-1].vector().array()
    
    def J(self,opt,ic,U,start,end):
        """
        The functional we want to minimize
        """
        n = len(U)
        timestep = (end-start)/float(n)
        s = 0
    
        s += 0.5*assemble(U[0]**2*dx)
        for i in range(n-2):
            s += assemble(U[i+1]**2*dx)
        s += 0.5*assemble(U[-1]**2*dx)
        return timestep*s
    def penalty_grad_J(self,P,loc,ic,m,h):
        """
        Gradient when we have penalty
        """
        xN = len(ic.vector().array())
        
        grad = np.zeros(m*xN)
        grad[:xN] = P[0][-1].vector().array().copy()[:]
        for i in range(m-1):
            grad[(i+1)*xN:(i+2)*xN] = project((P[i+1][-1]-P[i][0]),
                                              self.V).vector().array().copy()[:]
        
        return h*grad
        
if __name__ == "__main__":
    set_log_level(ERROR)

    mesh = UnitIntervalMesh(40)

    V = FunctionSpace(mesh,"CG",1)

    test1 = Burger1(V,mesh)

    opt = {'nu' : 0.1,'rhs':None}
    ic = project(Expression("x[0]*(1-x[0])"),V)
    start = 0
    end = 0.5
    Tn = 30
    
    test1.PDE_solver(ic,opt,start,end,Tn,show_plot=True)

    #test1.adjoint_solver(ic,opt,start,end,Tn)
    
    res = test1.solver(opt,ic,start,end,Tn)

    
    
    m=10
    my_l=False 
    res2 = test1.penalty_solver(opt,ic,start,end,Tn,m,[10],my_lbfgs=my_l)
    
    l = Function(V)
    
    l.vector()[:] = res2.x[:41]
    plot(l)
    interactive()
    
    
"""

class focp(FenicsOptimalControlProblem):

    def adjoint_ic(self,opt):
        raise NotImplementedError, 'adjoint_ic not implemented'

    def PDE_form(self,ic,opt,u,u_,v,timestep):
        raise NotImplementedError, 'PDE_form not implemented'

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        raise NotImplementedError, 'adjoint_form not implemented'

    def get_control(self,opt,ic,m):
        raise NotImplementedError, 'get_control not implemented'

    def get_opt(self,control,opt,ic,m):
        raise NotImplementedError, 'get_opt not implemented'

    def grad_J(self,P,opt,ic,h):
        raise NotImplementedError, 'grad_J not implemented'
    
    def J(self,opt,ic,U,start,end):
        raise NotImplementedError, 'J not implemented'

    def penalty_grad_J(self,P,opt,ic,m,h):
        raise NotImplementedError, 'penalty_grad_J not implemented'
"""
