from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros


    

H = [8,16,32,64]
P = [1,2,3,4]
for j in range(len(P)):
    print
    for i in range(len(H)):
        
        mesh = UnitSquareMesh(H[i],H[i])

        V = FunctionSpace(mesh,"Lagrange",P[j])
        u = TrialFunction(V)
        v = TestFunction(V)
        bc=DirichletBC(V,Constant(0),"on_boundary")
        u1 = Function(V)
        M, _ = assemble_system(inner(grad(u),grad(v))*dx,Constant(0)*v*dx,bc)
        
        #MM = matrix(M.array())
        
        nonz = M.nnz()
        N=len(u1.vector().array())**2
        print float(nonz)/sqrt(N),N,nonz,sqrt(N)

        
