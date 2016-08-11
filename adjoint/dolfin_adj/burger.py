from dolfin import *

def Dt(u,u_,timestep):
    return (u-u_)/timestep

def burger_solve(ic,start,end,V,Tn,show_plot=False):

    
    U = []
    u_ = ic.copy()
    U.append(u_.copy())
    if show_plot:
        plot(u_)
        interactive()

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
        if show_plot:
            plot(u_)
            interactive()

    return U


def J(U,T):
    n = len(U)
    timestep = T/float(n)
    s = 0
    for i in range(n):
        s += assemble(U[i]**2*dx)
    return timestep*s

def adjoint_burger_solve(ic,start,end,V,Tn):
    

    U = burger_solve(ic,start,end,V,Tn)
    P = []
    timestep = Constant((end-start)/float(Tn))

    nu = Constant(0.0001)
    
    p_ = project(Constant(0.0),V)
    P.append(p_.copy())

    p = Function(V)
    v = TestFunction(V)
    
    u = U[-1].copy()

    F = (Dt(p,p_,timestep)*v + u*p.dx(0)*v -nu*p.dx(0)*v.dx(0) +u*v)*dx
    
    bc = DirichletBC(V,0.0,"on_boundary")

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
    
if __name__ == "__main__":
    
    n = 40
    Tn = 30
    start = 0
    end = 0.5
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh,"CG",1)

    ic = project(Expression("x[0]*(1-x[0])"),V)

    U = burger_solve(ic,start,end,V,Tn,show_plot=False)

    print J(U,end-start)


    P = adjoint_burger_solve(ic,start,end,V,Tn)
    
    for i in range(len(P)):
        plot(P[i])
        interactive()
    """
    
    plot(ic + P[-1])
    interactive()
    """
