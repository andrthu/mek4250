from dolfin import *
from numpy import zeros, sqrt, pi,log,exp,array
from LeastSquare import LS


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


LS_h = []
LS_E = []
LS_L2 = []
LS_L1 = []
LS_Linf = []

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

        
        if i==0:
            z=1
            #plot(e)
            #interactive()
        
            
        E_LInf[i,j] = max(abs(u2.vector().array()-ue.vector().array()))
        E_L1[i,j] = assemble(abs(u-ue)*dx)                  
        E_L2[i,j] = errornorm(u,ue,'L2')
        E[i,j] = errornorm(u,ue,'H1')
        H2[i,j] = sqrt(Hp_sink(2+i,K[j]))


        if i!=0 or j!=2:
            LS_h.append(1./N[i])
            LS_E.append(E[i,j]/sqrt(Hp_sink(2,K[j])))
            
            LS_L2.append(E_L2[i,j]/sqrt(Hp_sink(2,K[j])))
            LS_L1.append(E_L1[i,j]/sqrt(Hp_sink(2,K[j])))
            LS_Linf.append(E_LInf[i,j]/sqrt(Hp_sink(2,K[j])))

            

CH = zeros((len(N),len(K)))
CL2 = zeros((len(N),len(K)))
CL1 = zeros((len(N),len(K)))
CLinf = zeros((len(N),len(K)))
for r in range(len(N)):
    CH[r,:] = 10**(r)*100*E[r,:]/H2[0,:]


    CL2[r,:] = (10**(r)*100)**2*E_L2[r,:]/H2[0,:]
    CL1[r,:] = (10**(r)*100)**2*E_L1[r,:]/H2[0,:]
    CLinf[r,:] = (10**(r)*100)**2*E_LInf[r,:]/H2[0,:]
for j in range(len(K)):
    print
    print 'k=%d' % K[j]
    print
    for i in range(len(N)):
        print 'N=%d gives following errors:' % N[i]
        print 'H1=%e, L2=%e, L1=%e, LInf=%e' % (E[i,j],E_L2[i,j], E_L1[i,j],E_LInf[i,j])
        
print    
print 'H1 constants for |u-ue|<ch|u''|'
print CH
print
print 'L2 constants for |u-ue|<ch**2|u''|'
print CL2
print
print 'L1 constants for |u-ue|<ch**2|u''|'
print CL1
print    
print 'LInf constants for |u-ue|<ch**2|u''|'
print CLinf
print
print H2



LS_h = array(LS_h)
LS_E = array(LS_E)

valH1 = LS(zeros(len(LS_h))+1,log(LS_h),log(LS_E))
valL2 = LS(zeros(len(LS_h))+1,log(LS_h),log(LS_L2))
valL1 = LS(zeros(len(LS_h))+1,log(LS_h),log(LS_L1))
valLinf = LS(zeros(len(LS_h))+1,log(LS_h),log(LS_Linf))

print
print "Least square results",len(LS_h)
print
print "H1: C=%f alpha=%f" % (exp(valH1[0,0]),valH1[1,0])
print
print "L2: C=%f alpha=%f" % (exp(valL2[0,0]),valL2[1,0])
print
print "L1: C=%f alpha=%f" % (exp(valL1[0,0]),valL1[1,0])
print
print "LInf: C=%f alpha=%f" % (exp(valLinf[0,0]),valLinf[1,0])
