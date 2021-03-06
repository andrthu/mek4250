from dolfin import *
from numpy import zeros, sqrt, pi
import time

def boundary(x,on_boundary): return on_boundary
N=[20,6]
P=[3,4]
K=10
for i in range(len(P)):
    start = time.time()
    
    mesh = UnitCubeMesh(N[i],N[i],N[i])
    V=FunctionSpace(mesh,'Lagrange',P[i])
    V2 = FunctionSpace(mesh,'Lagrange',P[i]+2)
    bc = DirichletBC(V,Constant(0),boundary)

    exact = Expression('sin(k*pi*x[0])*sin(k*pi*x[1])*sin(k*pi*x[2])',k=K)
    f = Expression('3*pow(k*pi,2)*sin(k*pi*x[0])*sin(k*pi*x[1])*sin(k*pi*x[2])',k=K)

    ue = interpolate(exact,V2)
    u=TrialFunction(V)
    v=TestFunction(V)

    a = inner(grad(u),grad(v))*dx
    L = v*f*dx

    u=Function(V)
    solve(a==L,u,bc,
          solver_parameters={"linear_solver": "cg"}) #hypre_amg
    e=u-ue
    A=assemble(e**2*dx)
    B = assemble(inner(grad(e),grad(e))*dx)
    end = time.time()
    print sqrt(A+B), (end-start)



