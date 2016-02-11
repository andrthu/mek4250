from dolfin import *
from numpy import zeros, sqrt, pi



def Hp_sink(p,k):
    return 0.5*(1-(k*pi)**(2*(p+1)))/(1-(k*pi)**2)
    

def boundary(x,on_boundary): return on_boundary

N = [100,1000,10000]
K=[1,10,100]


E = zeros((len(N),len(K)))
H2 = zeros((len(N),len(K)))
E_L2 = zeros((len(N),len(K)))
E_LInf = zeros((len(N),len(K)))
E_L1 = zeros((len(N),len(K)))


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
        u2 = interpolate(u,V2)
        e = u2-ue

        
        if i==2:
            z=1
            #plot(e)
            #interactive()
        
            
        E_LInf[i,j] = max(abs(u2.vector().array()-ue.vector().array()))
        E_L1[i,j] = assemble(abs(u-ue)*dx)                  
        E_L2[i,j] = errornorm(u,ue,'L2')
        E[i,j] = errornorm(u,ue,'H1')
        H2[i,j] = sqrt(Hp_sink(2+i,K[j]))

CH = zeros((len(N),len(K)))
CL2 = zeros((len(N),len(K)))
CL1 = zeros((len(N),len(K)))
CLinf = zeros((len(N),len(K)))
for r in range(len(N)):
    CH[r,:] = 10**(r)*100*E[r,:]/H2[0,:]


    CL2[r,:] = (10**(r)*100)**2*E_L2[r,:]/H2[0,:]
    CL1[r,:] = (10**(r)*100)**2*E_L1[r,:]/H2[0,:]
    CLinf[r,:] = (10**(r)*100)**2*E_LInf[r,:]/H2[0,:]

print E
print
print CH
print
print CL2
print
print CL1        
print 
print CLinf
