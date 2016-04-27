from dolfin import *
from scipy import linalg
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log,array

H=[8,16,32,64]
h_val = []
K = []
Ks = []
for i in range(len(H)):
    
    mesh = UnitIntervalMesh(H[i])

    V = FunctionSpace(mesh,'Lagrange',1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc=DirichletBC(V,Constant(0),"on_boundary")
    M, _ = assemble_system(u*v*dx,Constant(0)*v*dx,bc)
    M2,_ = assemble_system(inner(grad(u),grad(v))*dx,Constant(0)*v*dx,bc)

    l,v = linalg.eigh(matrix(M.array()))

    ls,vs = linalg.eigh(matrix(M2.array()))
                           
    l1 =float(max(l))
    l2 = min(l)

    ls1 =float(max(ls))
    ls2 = min(ls)
    
    h_val.append(mesh.hmax())
    K.append(l1/l2)
    Ks.append(ls1/ls2)
    
    print 'mass'
    print l1,l2,l1/l2
    print 'stiff'
    print ls1,ls2,ls1/ls2
print

Q = vstack([log(array(h_val)),ones(len(h_val))]).T
print 'mass'
print linalg.lstsq(Q, log(array(K)))[0]
print
print 'stiff'
print linalg.lstsq(Q, log(array(Ks)))[0]
print
h_val = []
K = []
Ks=[]
for i in range(len(H)):
    
    mesh = UnitSquareMesh(H[i]/2,H[i]/2)

    V = FunctionSpace(mesh,'Lagrange',1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc=DirichletBC(V,Constant(0),"on_boundary")
    M, _ = assemble_system(u*v*dx,Constant(0)*v*dx,bc)
    M2,_ = assemble_system(inner(grad(u),grad(v))*dx,Constant(0)*v*dx,bc)

    l,v = linalg.eigh(matrix(M.array()))
    ls,vs = linalg.eigh(matrix(M2.array()))

    l1 =float(max(l))
    l2 = min(l)

    ls1 =float(max(ls))
    ls2 = min(ls)

    h_val.append(mesh.hmax())
    K.append(l1/l2)
    Ks.append(ls1/ls2)
    
    print 'mass',
    print max(l),min(l),float(max(l))/min(l)
    print 'stiff'
    print ls1,ls2,ls1/ls2

print
Q = vstack([log(array(h_val)),ones(len(h_val))]).T
print 'mass'
print linalg.lstsq(Q, log(array(K)))[0]
print
print 'stiff'
print linalg.lstsq(Q, log(array(Ks)))[0]
print
h_val = []
K = []
Ks=[]
H=[16,32,64]
for i in range(len(H)):
    
    mesh = UnitCubeMesh(H[i]/8,H[i]/8,H[i]/8)

    V = FunctionSpace(mesh,'Lagrange',1)

    u = TrialFunction(V)
    v = TestFunction(V)
    bc=DirichletBC(V,Constant(0),"on_boundary")
    M, _ = assemble_system(u*v*dx,Constant(0)*v*dx,bc)
    M2,_ = assemble_system(inner(grad(u),grad(v))*dx,Constant(0)*v*dx,bc)

    l,v = linalg.eigh(matrix(M.array()))
    ls,vs = linalg.eigh(matrix(M2.array()))

    l1 =float(max(l))
    l2 = min(l)
    
    ls1 =float(max(ls))
    ls2 = min(ls)

    h_val.append(mesh.hmax())
    K.append(l1/l2)
    Ks.append(ls1/ls2)
    
    print 'mass'
    print max(l),min(l),float(max(l))/min(l)
    print 'stiff'
    print ls1,ls2,ls1/ls2
print

Q = vstack([log(array(h_val)),ones(len(h_val))]).T
print 'mass'
print linalg.lstsq(Q, log(array(K)))[0]
print
print 'stiff'
print linalg.lstsq(Q, log(array(Ks)))[0]
