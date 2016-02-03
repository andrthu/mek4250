from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros
from scipy import linalg, random

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
    
    
    a = random.rand(N+1)
    a[0]=0
    a[-1]=0

    
    f = Function(V)    
    f.vector()[:]=a
    x = matrix(f.vector().array())

    H1_norm = zeros(3)   
    L2_norm = zeros(3)
    Hm1_norm = zeros(2)

    #integration
    L2_norm[0] = sqrt(assemble(f**2*dx))
    H1_norm[0] = sqrt(assemble(inner(grad(f),grad(f))*dx))
    
    #matrix
    L2_norm[1] = sqrt(x*MM*x.T)
    H1_norm[1] = sqrt(x*AA*x.T)
    Hm1_norm[0] = sqrt(x*MM*linalg.inv(AA)*MM*x.T)
    
    #spectum

    l,v = linalg.eigh(AA,MM) #A*v=l*M*v
    v = matrix(v)
    l = matrix(diagflat(l))
    W = MM.dot(v)
    
    L2_norm[2] = sqrt(x*W*l**0*W.T*x.T)
    H1_norm[2] = sqrt(x*W*l*W.T*x.T)
    Hm1_norm[1] = sqrt(x*W*l**(-1)*W.T*x.T)
    
    print "N = %d" % N
    print "H1: " , H1_norm
    print
    print "L2: " , L2_norm
    print 
    print "H-1:" , Hm1_norm
    print 
    
    
