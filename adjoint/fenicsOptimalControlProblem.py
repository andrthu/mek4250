from dolfin import *
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.my_vector import SimpleVector

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
            N[i] = fraq + 1
        else:
            N[i] = fraq

        t = t + timestep*N[i]
        T[i+1] = t

    return N,T


class FenicsOptimalControlProblem():

    def __init__(self,V,mesh):

        self.V = V
        self.mesh = mesh
        

        self.Lbfgs_options = self.default_Lbfgs_options()
        

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

    def penalty_J(self,opt,ic,U,start,end,tn,m,mu):

        
        N,T=partition(start,end,Tn,m)

        s = 0
        for i in range(m):
            s += self.J(opt,U[i][0],T[i],T[i+1])
        penalty = 0
        for i in range(len(U)-1):
            penalty = penalty +0.5*assemble((U[i][-1]-U[i+1][0])**2*dx)
    
        return s + mu*penalty

                
        
    def PDE_solver(self,ic,opt,start,end,Tn,show_plot=False):

        U = []

        u_ = ic.copy()
        if show_plot:
            plot(u_)
        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)

        timestep = Constant((end-start)/float(Tn))

        F,bc = self.PDE_form(ic,opt,u,u_,v,timestep)


        t  = start

        while (t<end - DOLFIN_EPS):
            solve(F==0,u,bc)
            u_.assign(u)
            U.append(u_.copy())

            t += float(timestep)

            if show_plot:
                plot(u_)

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

        
    def adjoint_solver(self,ic,opti,start,end,Tn):

        U = self.PDE_solver(ic,opt,start,end,Tn)
        p_ic = self.adjoint_ic(opt)
        
        return self.adjoint_interval_solver(opt,p_ic,U,start,end,Tn)


    def penalty_PDE_solver(self,opt,ic,lam_ic,start,end,Tn,m):

        N,T = time_partition(start,end,Tn,m)


        U = []
        V = self.V

        U.append(self.PDE_solver(ic,opt,T[0],T[1],V,N[0]))
        for i in range(1,m):
            U.append(self.PDE_solver(lam_ic[i-1],opt,T[i],T[i+1],N[i]))

        return U

    def penalty_adjoint_solver(self,ic,lam_ic,opti,start,end,Tn,m,mu):

        N,T = time_partition(start,end,Tn,m)
    
        U =  general_burger_solver(ic,lam_ic,start,end,V,Tn,m)
        
        P = []
    
        for i in range(m-1):
            P_ic = project((U[i][-1]-lam_ic[i])*Constant(mu),V)
            sta = T[i]
            en = T[i+1]
            P.append(self.adjoint_interval_solver(opt,P_ic,U[i],sta,en,N[i]))

        P_ic = self.adjoint_ic(opt)
        P.append(interval_adjoint(optP_ic,U[-1],T[-2],T[-1],N[-1]))

        return P

    def solver(self,opt,ic,start,end,Tn,Lbfgs_options=None):
        h = self.mesh.hmax()
        
        def J(x):
            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            U = self.PDE_solver(loc_ic,loc_opt,start,end,Tn)
            return self.J(loc_opt,loc_ic,U,start,end)
        
        def grad_J(x):

            loc_opt,loc_ic = self.get_opt(x,opt,ic,1)
            
            P = self.adjoint_solver(loc_ic,loc_opt,start,end,Tn)

            return self.grad_J(P,loc_opt,loc_ic,h)


        control0 = SimpleVector(self.get_control(opt,ic,1))

        if Lbfgs_options==None:
            Loptions = self.Lbfgs_options
        else:
            Loptions = self.Lbfgs_options
            for key, val in Lbfgs_options.iteritems():
                Loptions[key]=val


        solver = Lbfgs(J,grad_J,control0,options=Loptions)

        res = solver.solve()
        x = Function(self.V)

        x.vector()[:] = res['control'].array()
        plot(x)
        interactive()
        return res


    def penalty_solver(self,opt,ic,start,end,Tn,m,mu_list,Lbfgs_options=None):

        h = self.mesh.hmax()
        X = Function(self.V)
        xN = len(C.vector().array())
        

        for i in range(len(mu_list)):
            def J(x):
            
                cont_e = len(x)-(m-1)*xN 
                loc_opt,loc_ic = self.get_opt(x[:cont_e],opt,ic,m)

                lam = []
                lam.append(loc_ic)
                for i in range(m-1):
                    l = Function(self.V)
                    l.vector()[:] = x.copy()[cont_e+i*xN:cont_e+(i+1)*xN]
                    lam.append(l.copy())
            
            

                U = self.penalty_PDE_solver(loc_opt,loc_ic,lam_ic,start,end,Tn,m):
                return self.penalty_J(loc_opt,loc_ic,U,start,end,tn,m,mu_list[i])
        
            def grad_J(x):

                cont_e = len(x)-(m-1)*xN 
                lopt,lic = self.get_opt(x[:cont_e],opt,ic,m)
            
                lam = []
                lam.append(loc_ic)
                for i in range(m-1):
                    l = Function(self.V)
                    l.vector()[:] = x.copy()[cont_e+i*xN:cont_e+(i+1)*xN]
                    lam.append(l.copy())
                mu = mu_list[i]
                P=self.penalty_adjoint_solver(lic,lam,lopt,start,end,Tn,m,mu)

                return self.penalty_grad_J(P,lopt,lic,h)


            control0 = SimpleVector(self.get_control(opt,ic,m))

            if Lbfgs_options==None:
                Loptions = self.Lbfgs_options
            else:
                Loptions = self.Lbfgs_options
                for key, val in Lbfgs_options.iteritems():
                    Loptions[key]=val


            solver = Lbfgs(J,grad_J,control0,options=Loptions)

            res = solver.solve()


class Burger1(FenicsOptimalControlProblem):
    
    #opt = {nu : ...}
    def adjoint_ic(self,opt):
        return project(Constant(0.0),self.V)

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

    def PDE_form(self,ic,opt,u,u_,v,timestep):
        nu = Constant(opt['nu'])
        F = (self.Dt(u,u_,timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")
        return F,bc

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        nu = Constant(opt['nu'])

        F = -(-self.Dt(p,p_,timestep)*v + u*p.dx(0)*v -nu*p.dx(0)*v.dx(0) +2*u*v)*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")
        return F,bc

    def get_control(self,opt,ic,m):
        if m==1:
            return ic.copy().vector().array()
        
        N = len(ic.vector().array())

        x = np.zeros(m*N)

        x[:N] = ic.copy().vector().array()

        for i in range(m-1):

            x[(i+1)*N:(i+2)
        
    def get_opt(self,control,opt,ic,m):

        g = Function(self.V)

        g.vector()[:] = control[:]

        return opt,g

    def grad_J(self,P,opt,ic,h):
        return h*P[-1].vector().array()
    
    def J(self,opt,ic,U,start,end):
        n = len(U)
        timestep = (end-start)/float(n)
        s = 0
    
        s += 0.5*assemble(U[0]**2*dx)
        for i in range(n-2):
            s += assemble(U[i+1]**2*dx)
        s += 0.5*assemble(U[-1]**2*dx)
        return timestep*s
        
        
if __name__ == "__main__":
    set_log_level(ERROR)

    mesh = UnitIntervalMesh(40)

    V = FunctionSpace(mesh,"CG",1)

    test1 = Burger1(V,mesh)

    opt = {'nu' : 0.0001}
    ic = project(Expression("x[0]*(1-x[0])"),V)
    start = 0
    end = 0.5
    Tn = 30
    
    #test1.PDE_solver(ic,opt,start,end,Tn,show_plot=True)

    #test1.adjoint_solver(ic,opt,start,end,Tn)

    res = test1.solver(opt,ic,start,end,Tn)

    print res['iteration']
