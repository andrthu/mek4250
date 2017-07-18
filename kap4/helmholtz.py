from dolfin import *
from numpy import zeros, sqrt, pi,matrix



def helm_solve(c,N,f,p=1):
    
    
    #mesh = UnitIntervalMesh(N)
    mesh=UnitSquareMesh(N,N)
    V=FunctionSpace(mesh,'Lagrange',p)
    V2 = FunctionSpace(mesh,'Lagrange',p+2)
    #bc = DirichletBC(V,Constant(0),boundary)

    
    
    
    u=TrialFunction(V)
    v=TestFunction(V)


    a = (inner(grad(u),grad(v))+c*u*v)*dx

    L = v*f*dx


    u=Function(V)
    solve(a==L,u)

    plot(u)
    interactive()

if __name__=="__main__":

    N = 100
    c = 0.1
    f = Expression('sin(2*pi*x[0])+cos(x[1]*pi*3)')

    helm_solve(c,N,f,p=1)
    
