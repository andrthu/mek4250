from dolfin import *

def Dt(u,u_,timestep):
    return (u-u_)/timestep

def burger_solve(ic,start,end,V,Tn):

    
    U = []
    u_ = ic.copy()
    U.append(u_.copy())

    u = Function(V)
    v = TestFunction(V)
    
    
    timestep = Constant((end-start)/float(Tn))

    nu = Constant(0.0001)

    F = (Dt(u,u_,timestep)*v + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V,0.0,"on_boundary")

    t  = start

    while (t<end - DOLFIN_EPS):
        solve(F==0,u,bc)
        u_.assign(u)
        U.append(u_.copy())

        t += float(timestep)

    return U


def J(U,T):
    n = len(U)
    timestep = T/float(n)
    s = 0
    for i in range(n):
        s += assemble(U[i]**2*dx)
    return timestep*s

def adjoint_burger_solve(ic,start,end,V,Tn):
    return "nix"

if __name__ == "__main__":
    
    n = 10
    Tn = 10
    start = 0
    end = 1
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh,"CG",1)

    ic = project(Expression("10*sin(2*pi*x[0])"),V)

    U = burger_solve(ic,start,end,V,Tn)

    print J(U,end-start)
