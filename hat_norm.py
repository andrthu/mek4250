from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros
from scipy import linalg

def boundary(x,on_boundary): return on_boundary

N_val=[10,100,1000]

for N in N_val:
    mesh = UnitIntervalMesh(N)
    V = FunctionSpace(mesh,"Lagrange",1)
    u = TrialFunction(V)
    v = TestFunction(V)
    bc=DirichletBC(V,Constant(0),boundary)

    A, _ = assemble_system(inner(grad(u),grad(v))*dx,Constant(0)*v*dx,bc)
    M, _ = assemble_system(u*v*dx,Constant(0)*v*dx,bc)
    AA = matrix(A.array())
    MM= matrix(M.array())

    a = zeros(N+1)
    a[N/2] = 1
    f = Function(V)
    f.vector()[:]=a
    x = matrix(f.vector().array())
    
    L2 = sqrt(x*MM*x.T)
    H1 = sqrt(x*AA*x.T)
    Hm1 = sqrt(x*MM*linalg.inv(AA)*MM*x.T)
    print "N=%d: H1=%f L2=%f H-1=%f" % (N,H1,L2,Hm1)
    

