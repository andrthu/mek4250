from dolfin import *
from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize as Mini
import numpy as np
from fenicsOptimalControlProblem import FenicsOptimalControlProblem


class HeatControl(FenicsOptimalControlProblem):
    
    #opt = {'rhs':f, 'c':c.'uT':uT}
    
    def adjoint_ic(self,opt,U):
        return project((U[-1]-opt['uT']),self.V)

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

        

    def PDE_form(self,ic,opt,u,u_,v,rhs,timestep):

        c = Constant(opt['c'])
        

        F = (self.Dt(u,u_,timestep)*v + c*u.dx(0)*v.dx(0)-rhs*v)*dx
        bc = DirichletBC(self.V,0.0,"on_boundary")

        return F,bc
        
        

    def adjoint_form(self,opt,u,p,p_,v,timestep):
        c = Constant(opt['c'])

        F = -(-self.Dt(p,p_,timestep)*v - c*p.dx(0)*v.dx(0))*dx

        bc = DirichletBC(self.V,0.0,"on_boundary")

        return F,bc


    def get_control(self,opt,ic,m):
        
        N = len(ic.vector().array())
        f = opt['rhs']
        Nt = len(f)
        x = np.zeros(N*Nt+(m-1)*N)
        
        for i in range(Nt):
            x[i*N:(i+1)*N] = f[i].copy().vector().array()[:]
        return x

    def get_opt(self,control,opt,ic,m):
        
        f = opt['rhs']
        Nt = len(f)
        N = len(ic.vector().array())

        for i in range(Nt):
            f[i].vector()[:] = control[i*N:(i+1)*N]
            

        return opt,ic
        

    def grad_J(self,P,opt,ic,h):
        
        f = opt['rhs']
        N = len(ic.vector().array())
        Nt = len(f)
        x = np.zeros(N*Nt)
        timestep = opt['T']/float(len(f))
        #x[0:N] = f[0].copy().vector().array()[:]
        for i in range(0,Nt):
            x[i*N:(i+1)*N]=timestep*(P[-(i+1)].copy().vector().array()[:] + f[i].copy().vector().array()[:])

        return h*x
        
    
    def J(self,opt,ic,U,start,end):
        n = len(U)
        timestep = (end-start)/float(n)
        s = 0
        
        f = opt['rhs']
        s += 0.5*assemble(f[0]**2*dx)
        for i in range(n-2):
            s += assemble(f[i+1]**2*dx)
        s += 0.5*assemble(f[-1]**2*dx)

        s2 = 0.5*assemble((U[-1]-opt['uT'])**2*dx)
        return timestep*s +s2

    def penalty_grad_J(self,P,opt,ic,m,h):
        f = opt['rhs']
        N = len(ic.vector().array())
        Nt = len(f)
        x = np.zeros(N*Nt +(m-1)*N )
        timestep = opt['T']/float(len(f))
        #x[0:N] = f[0].copy().vector().array()[:]
        p = self.gather_penalty_funcs(P)
        for i in range(0,Nt):
            x[i*N:(i+1)*N]=timestep*(p[-(i+1)].copy().vector().array()[:] + f[i].copy().vector().array()[:])
        
        for i in range(m-1):
            x[Nt*N+ i*N:Nt*N+ (i+1)*N] =project((P[i+1][-1]-P[i][0]),
                                              self.V).vector().array().copy()[:]
        return h*x
        
    def rhs_finder(self,Rhs,rhs,i):
        rhs.assign(Rhs[i])


if __name__== '__main__':
    import time
    set_log_level(ERROR)
    mesh = UnitIntervalMesh(20)

    V = FunctionSpace(mesh,"CG",1)

    test1 = HeatControl(V,mesh)
    
    ic = project(Constant(0.0),V)
    start = 0
    end = 0.5
    Tn = 30
    RHS = []
    m = 3
    r = Expression('sin(pi*x[0])')
    for i in range(Tn+1):
        RHS.append(project(r,V))

    ut = Constant(0.0)
    
    opt = {'c' : 0.1,'rhs':RHS,'uT':project(ut,V),'T':end-start}

    #test1.PDE_solver(ic,opt,start,end,Tn,show_plot=True)
    
    #res = test1.solver(opt,ic,start,end,Tn)
    res = test1.penalty_solver(opt,ic,start,end,Tn,m,[1])
    N = len(ic.vector().array())
    #print res
    i = 0
    """
    while i <Tn:
        f = Function(V)
        
        f.vector()[:] = res.x[i*N:(i+1)*N]
        plot(f)
        time.sleep(1)
        i+=1
    
    """
