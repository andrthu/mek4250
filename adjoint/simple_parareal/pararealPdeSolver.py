from dolfin import *
import numpy as np
import time
from partition_funcs import int_par_len

def rhs_part(N,M,i):

    f = N/M
    r = N%M

    s = 0

    for j in range(i):
        if r - j >0:
            s += f +1
        else:
            s += f
    return s

def gather_u(u):
    """
    Gathers the u lists into one list U
    """
    U = []
    
    for j in range(len(u[0])):
        U.append(u[0][j])
    
    for i in range(len(u)-1):
        
        for j in range(1,len(u[i])):
            U.append(u[i][j])
        
    return U

class PararealPdeSolver():

    def __init__(self,V,mesh):
        self.V = V
        self.mesh = mesh

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

        
    def PDE_form(self,u,u_,v,r,dt):
        
        return (self.Dt(u,u_,dt)*v + inner(grad(u),grad(v))-r*v)*dx 
        
        

    def implicit_solver(self,ic,rhs,bc,dt,N):

        U = []

        u_ = ic.copy()

        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)
        r = rhs[0]
        F = self.PDE_form(u,u_,v,r,dt)
        
        for i in range(N):
            r.assign(rhs[i])
            #plot(u_)
            solve(F==0,u,bc)
            u_.assign(u)
            U.append(u_.copy())
            #time.sleep(1)
        interactive()
        return U

    def propagator_iteration(self,u,c_u,rhs,bc,N,M,dt,dT):

        S = []
        S.append(project(Constant(0.0),self.V))
        for i in range(M):
            print i,len(u[i])
            
            S.append(project((u[i][-1].copy()-c_u[i+1].copy())*Constant(1./dT),self.V))

        delta = self.implicit_solver(project(Constant(0.0),self.V),S,bc,dT,M)
        for i in range(M):
            c_u[i+1] = project(u[i][-1] + delta[i+1],self.V)

        u = []
        
        
        for i in range(M):
            u.append(self.implicit_solver(c_u[i],rhs[rhs_part(N,M,i):-1],bc,dt,
                                          int_par_len(N+1,M,i)-1))
        return u,c_u

    def parareal_order_solver(self,ic,rhs,bc,dt,dT,N,M,order=3):
        
        coarse_rhs = []
        for i in range(M):
            coarse_rhs.append(rhs[rhs_part(N,M,i)])
        coarse_u = self.implicit_solver(ic,coarse_rhs,bc,dT,M)

        u = []
        for i in range(M):
            u.append(self.implicit_solver(coarse_u[i],rhs[rhs_part(N,M,i):-1],
                                          bc,dt,int_par_len(N+1,M,i)-1))

        for k in range(order-1):
            u,coarse_u = self.propagator_iteration(u,coarse_u,rhs,
                                                   bc,N,M,dt,dT)

        return gather_u(u)


if __name__ == "__main__":
    #set_log_level(ERROR)
    mesh = UnitIntervalMesh(10)

    V = FunctionSpace(mesh,"CG",1)

    test = PararealPdeSolver(V,mesh)

    N = 100
    M = 3
    rhs = []
    for i in range(N+1):
        rhs.append(project(Constant(0.0),V))

    bc = DirichletBC(V,0.0,"on_boundary")

    dt = 1./N
    dT = 1./M
    ic = project(Expression('x[0]*(1-x[0])'),V)
    #test.implicit_solver(ic,rhs,bc,dt,N)
    U = test.parareal_order_solver(ic,rhs,bc,dt,dT,N,M)
