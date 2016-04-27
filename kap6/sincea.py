from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log
from scipy import linalg, random

N = [10,100,1000,10000]
alpha_val = [1,1.0e-2]

h_val = zeros(len(N))
E_val = zeros((len(alpha_val),len(N)))
cea =[]

for i in range(len(alpha_val)):
    for j in range(len(N)):

        mesh=UnitIntervalMesh(N[j])

        V=FunctionSpace(mesh,'Lagrange',1)
        V2=FunctionSpace(mesh,'Lagrange',1+3)
        u = TrialFunction(V)
        v = TestFunction(V)

        
        alpha = Constant(alpha_val[i])

        f = Expression('%e*pi*pi*sin(pi*x[0])-pi*cos(pi*x[0])'%alpha_val[i])

        

        a = (-u.dx(0)*v + alpha*inner(grad(u),grad(v)))*dx
        L=f*v*dx
        
        
        

        ue=Expression('sin(pi*x[0])')

        def boundary(x,on_boandary):
            return on_boandary

        bc = DirichletBC(V,ue,boundary)
        
        U = Function(V)

        solve(a==L,U,bc)
        
        Ue = project(ue,V2)

        E_val[i][j] = errornorm(U,Ue,'H1')
        h_val[j] = mesh.hmin()


        
        plot(U)
        interactive()
    

    Q = vstack([log(h_val),ones(len(N))]).T

    cea.append(linalg.lstsq(Q, log(E_val[i]))[0])

    

print E_val
print cea

    
    
