from dolfin import *
from numpy import zeros, sqrt, pi,matrix
import time

def boundary(x,on_boundary): return on_boundary
#N=[80,60,40,22,9,7]
#P=[1,2,3,4,5,6]
P=[1]
N=[10]
K=1
for i in range(len(P)):
    
    start = time.time()
    
    mesh = UnitIntervalMesh(N[i])#,N[i])
    V=FunctionSpace(mesh,'Lagrange',P[i])
    V2 = FunctionSpace(mesh,'Lagrange',P[i]+2)
    bc = DirichletBC(V,Constant(0),boundary)

    exact = Expression('sin(k*pi*x[0])*sin(k*pi*x[1])',k=K)
    f = Expression('2*pow(k*pi,2)*sin(k*pi*x[0])*sin(k*pi*x[1])',k=K)

    ue = interpolate(exact,V2)
    u=TrialFunction(V)
    v=TestFunction(V)

    a = inner(grad(u),grad(v))*dx
    L = v*f*dx

    a2 = u*v*dx
    h = mesh.hmin()

    A,b =assemble_system(a,L,bc)
    A2,b2 = assemble_system(a2,L,bc)
    print A.array()#.dot((zeros(36)+1))
    print len(A.array())
    print 60*A2.array() 
    u=Function(V)
    solve(a==L,u,bc,
          solver_parameters={"linear_solver": "cg"}) #hypre_amg
    e=u-ue
    A=assemble(e**2*dx)
    B = assemble(inner(grad(e),grad(e))*dx)
    end = time.time()
    print sqrt(A+B), (end-start)
    print h

