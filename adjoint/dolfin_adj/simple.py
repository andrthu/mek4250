from dolfin import *
from scipy.optimize import minimize
import numpy as np
#from dolfin_adjoint import *


def ODE_solver(y0,N,U,show_plot=False):

    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)

    y = TrialFunction(V)
    v = TestFunction(V)

    #u = Function(V)
    #u.vector().array()[:]=U[:]
    u = interpolate(U,V)

    a = (y.dx(0)*v - v*y)*dx
    L = u*v*dx

    ic = y0
    bc = DirichletBC(V, Constant(ic),"on_boundary && near(x[0],0.0)")

    Y = Function(V)
    #A,b=assemble_system(a,L,bc)
    solve(a==L,Y,bc)
    if show_plot:
        plot(Y)
        interactive()
    return Y

def adjoint_ODE_solver(y0,yT,N,u):
    
    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)
    y = ODE_solver(y0,N,u)

    l = TrialFunction(V)
    v = TestFunction(V)

    a = (-l.dx(0)*v -v*l)*dx
    L = Constant(0.0)*v*dx

    ic = y.vector().array()[0]-yT

    bc = DirichletBC(V, Constant(ic),"on_boundary && near(x[0],1.0)")
    lam = Function(V)

    solve(a==L,lam,bc)

    plot(lam)
    interactive()
    

def J_Func(u,y0,yT,N):
    
    y = ODE_solver(y0,N,u,show_plot=False)

    A = assemble(u**2*dx)

    return 0.5*(A + (y.vector().array()[0]-yT)**2)

def opti(y0,yT,N):

    mesh = UnitIntervalMesh(N)
    dt = mesh.hmax()
    
    def J(u):
        return J_Func(u,y0,yT,N)

    def grad_J(u):

        lam = adjoint_ODE_solver(y0,yT,N,u)

        gr = u + lam

        return gr.vector().array()*dt
        
if __name__ == "__main__":
    
    y0=1
    yT=1
    N=100
    #u = np.zeros(N+1)
    u= Constant(0.0)
    y = ODE_solver(y0,N,u,show_plot=True)

    adjoint_ODE_solver(y0,yT,N,u)
