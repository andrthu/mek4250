from dolfin import *
from numpy import zeros, sqrt, pi



def Hp_sink(p,k):
    return 0.5*(1-(k*pi)**(2*(p+1)))/(1-(k*pi)**2)
    

def boundary(x,on_boundary): return on_boundary

N = [100,1000,10000,100000]
K=[1,10,100,1000]
E = zeros((len(N),len(K)))
H2 = zeros((len(N),len(K)))

for i in range(len(N)):

    

    mesh = UnitIntervalMesh(N[i])
    V = FunctionSpace(mesh,'Lagrange',1)
    V2 = FunctionSpace(mesh,'Lagrange',1+3)
    
    bc = DirichletBC(V,Constant(0),boundary)
    for j in range(len(K)):
        f = Expression('(k*pi*pi*k)*sin(k*pi*x[0])',k=K[j])
        ue1 = Expression('sin(k*pi*x[0])',k=K[j])
        ue = interpolate(ue1,V2)

        u = TrialFunction(V)
        v = TestFunction(V)

        a = inner(grad(u),grad(v))*dx
        L = f*v*dx
    
        u=Function(V)
        solve(a==L,u,bc)
        
        if i==2:
            e=u-ue
            A = assemble(e**2*dx)
            B = assemble(inner(grad(e),grad(e))*dx)
            print sqrt(A+B)
            #plot(e)
            #interactive()
                        
        
        E[i,j] = errornorm(u,ue,'H1')
        H2[i,j] = sqrt(Hp_sink(2+i,K[j]))

C = zeros((len(N),len(K)))

C[0,:] = 100*E[0,:]/H2[0,:]
C[1,:] = 1000*E[1,:]/H2[0,:]
C[2,:] = 10000*E[2,:]/H2[0,:]
C[3,:] = 100000*E[3,:]/H2[0,:]

print E
print 
print H2
print
print C
        
