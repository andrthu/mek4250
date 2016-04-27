from dolfin import *
from scipy import linalg
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log,array

H = [8,16,32,64]

P=[1,2,3]

for i in range(len(P)):
    K=[]
    h_val=[]
    for j in range(len(H)):
        

        mesh = UnitIntervalMesh(H[j])

        V = FunctionSpace(mesh,'Lagrange',P[i])

        u = TrialFunction(V)
        v = TestFunction(V)
        bc=DirichletBC(V,Constant(0),"on_boundary")
        M, _ = assemble_system(u*v*dx,Constant(0)*v*dx,bc)
        
        l,v = linalg.eigh(matrix(M.array()))
    
        l1 =float(max(l))
        l2 = min(l)
        h_val.append(mesh.hmax())
        K.append(l1/l2)

    Q = vstack([log(array(h_val)),ones(len(h_val))]).T
    print linalg.lstsq(Q, log(array(K)))[0]
