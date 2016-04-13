from dolfin import *
import time
def u0_boundary(x, on_boundary):
    return on_boundary
for N in [5]:
    
    mesh = UnitSquareMesh(N,N)
    q=5
    V = FunctionSpace(mesh,'Lagrange',q)
    V2 =FunctionSpace(mesh,'Lagrange',q+3)
    f = Expression('2*pow(pi,2)*sin(pi*x[0])*sin(pi*x[1])')
    u0=Constant(0)
    e = Expression('sin(pi*x[0])*sin(pi*x[1])')
    
    v = TestFunction(V)
    u = TrialFunction(V)

    A = inner(nabla_grad(u),nabla_grad(v))*dx
    L = f*v*dx

    bc = DirichletBC(V,u0,u0_boundary)
    u=Function(V)
    t0=time.time()
    solve(A==L,u,bc)
    print "time",time.time()-t0
    
    
    print "error",errornorm(e,u)
    plot(u)
    interactive()
