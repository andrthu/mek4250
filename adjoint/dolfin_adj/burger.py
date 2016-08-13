from dolfin import *
from scipy.optimize import minimize as Mini
import numpy as np

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
    
    s += 0.5*assemble(U[0]**2*dx)
    for i in range(n-2):
        s += assemble(U[i+1]**2*dx)
    s += 0.5*assemble(U[-1]**2*dx)
    return timestep*s

def interval_adjoint(p_ic,U,start,end,V,Tn):
    P = []
    timestep = Constant((end-start)/float(Tn))

    nu = Constant(0.0001)
    
    p_ = p_ic.copy()
    P.append(p_.copy())

    p = Function(V)
    v = TestFunction(V)
    
    u = U[-1].copy()

    F = -(Dt(p,p_,timestep)*v + u*p.dx(0)*v -nu*p.dx(0)*v.dx(0) +2*u*v)*dx
    
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
    

def adjoint_burger_solve(ic,start,end,V,Tn):
    
    
    U = burger_solve(ic,start,end,V,Tn)
    p_ic = project(Constant(0.0),V)
    return interval_adjoint(p_ic,U,start,end,V,Tn)
    """
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
    """
def opti(ic,start,end,V,Tn,mesh):
    
    h = mesh.hmax()

    def red_J(g):
        G = Function(V)
        G.vector()[:]=g.copy()[:]
        U = burger_solve(G,start,end,V,Tn)
        return J(U,end-start)

    def grad_J(g):
        G = Function(V)
        G.vector()[:]=g.copy()[:]
        P = adjoint_burger_solve(G,start,end,V,Tn)

        return -h*P[-1].vector().array().copy()
    
    
    

    res = Mini(red_J,ic.vector().array().copy(),method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': True})

    X = Function(V)
    X.vector()[:] = res.x.copy()[:]

    plot(X)
    interactive()

def double_J(U,T,mu):
    
    timestep = T/float((len(U[0])+len(U[1])-1))
    s = 0
    s += 0.5*assemble(U[0][0]**2*dx)
    for i in range(len(U[0])-2):
        s += assemble(U[0][i+1]**2*dx)
    s += 0.5*assemble(U[0][-1]**2*dx)

    s += 0.5*assemble(U[1][0]**2*dx)
    for i in range(len(U[1])-2):
        s += assemble(U[1][i+1]**2*dx)
    s += 0.5*assemble(U[1][-1]**2*dx)

    penalty = 0.5*mu*assemble((U[0][-1]-U[1][0])**2*dx)
    
    return timestep*s + penalty

def double_burger_solver(ic,lam_ic,start,end,V,Tn,show_plot=False):

    
    
    T = end-start
    mid = start + (Tn/2+Tn%2)*(T/float(Tn))
    
    U1 = burger_solve(ic,start,mid,V,Tn/2 +Tn%2,show_plot=show_plot)    
    U2 = burger_solve(lam_ic,mid,end,V,Tn/2,show_plot=show_plot)

    return [U1,U2]

def double_adjoint_burger_solve(ic,lam_ic,start,end,V,Tn,mu):

    U = double_burger_solver(ic,lam_ic,start,end,V,Tn,show_plot=False)
    T = end-start
    mid = start + (Tn/2+Tn%2)*(T/float(Tn))
    print
    print start,end,mid
    print
    p1_ic = project((U[0][-1]-lam_ic)*Constant(mu),V)
    #plot(p1_ic)
    #interactive()
    P1 = interval_adjoint(p1_ic,U[0],start,mid,V,len(U[0])-1)
    P2 = interval_adjoint(project(Constant(0.0),V),U[1],mid,end,V,len(U[1])-1)
    return [P1,P2]
    

def double_opti(ic,start,end,V,Tn,mesh,mu):
    
    h = mesh.hmax()

    def red_J(x):
        G = Function(V)
        lam = Function(V)
        xN = len(x)
        G.vector()[:] = x.copy()[:xN/2]
        lam.vector()[:] = x.copy()[xN/2:]

        U = double_burger_solver(G,lam,start,end,V,Tn)
        return double_J(U,end-start,mu)

    def grad_J(x):
        G = Function(V)
        lam = Function(V)
        xN = len(x)
        G.vector()[:] = x.copy()[:xN/2]
        lam.vector()[:] = x.copy()[xN/2:]

        P = double_adjoint_burger_solve(G,lam,start,end,V,Tn,mu)
        
        grad = x.copy()
        grad[:xN/2] = -P[0][-1].vector().array().copy()[:]
        grad[xN/2:] = project((-P[1][-1]-P[0][0]),V).vector().array().copy()[:]
        return h*grad
    
    icN = len(ic.vector().array())
    init = np.zeros(2*icN)

    init[:icN] = ic.vector().array().copy()[:]

    res = Mini(red_J,init,method='L-BFGS-B', jac=grad_J,
               options={'gtol': 1e-6, 'disp': True})

    X = Function(V)
    Y = Function(V)
    X.vector()[:] = res.x.copy()[:icN]
    Y.vector()[:] = res.x.copy()[icN:]
    
    plot(X)
    interactive()
    plot(Y)
    interactive()
    
if __name__ == "__main__":
    
    n = 40
    Tn = 30
    start = 0
    end = 0.5
    mu = 1
    mesh = UnitIntervalMesh(n)
    V = FunctionSpace(mesh,"CG",1)

    ic = project(Expression("x[0]*(1-x[0])"),V)
    lam_ic = project(Constant(0.0),V)
    #U = burger_solve(ic,start,end,V,Tn,show_plot=True)

    


    #P = adjoint_burger_solve(ic,start,end,V,Tn)
    
    #D_U = double_burger_solver(ic,lam_ic,start,end,V,Tn,show_plot=True)
    D_P = double_adjoint_burger_solve(ic,lam_ic,start,end,V,Tn,mu)
    """
    for k in range(2):
        for i in range(len(D_P[k])):
            plot(D_P[k][i])
            interactive()
    """
    """
    for i in range(len(P)):
        plot(P[i])
        interactive()
    """
    
    #plot(ic + P[-1])
    #interactive()

    #opti(ic,start,end,V,Tn,mesh)
    double_opti(ic,start,end,V,Tn,mesh,mu)
    #print J(U,end-start)
    #print len(U),len(P)
    
