from dolfin import *
#from dolfin_adjoint import *


def ODE_solver(y0,N,u,show_plot=False):

    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'CG',1)

    y = TrialFunction(V)
    v = TestFunction(V)

    u = interpolate(u,V)

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
    
    
if __name__ == "__main__":
    u = Constant(0.0)
    y0=1
    yT=1
    N=100
    
    y = ODE_solver(y0,N,u,show_plot=True)

    adjoint_ODE_solver(y0,yT,N,u)
