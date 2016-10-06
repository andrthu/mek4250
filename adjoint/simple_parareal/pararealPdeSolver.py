from dolfin import *
import numpy as np


class PararealPdeSolver():

    def __init__(self,V,mesh):
        self.V = V
        self.mesh = mesh

    def Dt(self,u,u_,timestep):
        return (u-u_)/timestep

        
    def PDE_form(self,u,u_,v,dt):
        
        return (self.Dt(u,u_,dt)*v + inner(grad(u),grad(v)))*dx 
        
        

    def implicit_solver(self,ic,rhs,bc,dt,N):

        U = []

        u_ = ic.copy()

        U.append(u_.copy())

        u = Function(self.V)
        v = TestFunction(self.V)
        
        a = self.PDE_form(u,u_,v,dt)
        
        for i in range(N):
            plot(u_)
            L = rhs[i]*v*dx
            solve(L==a,u,bc)
            u_.assign(u)
            U.append(u_.copy())
            
        return U

if __name__ == "__main__":
    
    mesh = UnitIntervalMesh(40)

    V = FunctionSpace(mesh,"CG",1)


    
    
    test = PararealPdeSolver(V,mesh)

    N = 10

    rhs = []
    for i in range(N+1):
        rhs.append(project(Constant(0.0),V))

    bc = DirichletBC(V,0.0,"on_boundary")

    dt = 1./N

    ic = project(Expression('x[0]*(1-x[0])'),V)
    test.implicit_solver(ic,rhs,bc,dt,N)
