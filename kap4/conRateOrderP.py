from dolfin import *
from numpy import zeros, sqrt, pi,log,exp,array
from LeastSquare import LS


def Hp_sink(p,k):
    return 0.5*(1-(k*pi)**(2*(p+1)))/(1-(k*pi)**2)
    

def boundary(x,on_boundary): return on_boundary


def findRate(p,K,N):
    

    E_L1 = []
    
    E_Linf = []

    E = []

    E_L2 = []


    LS_h = []
    


    for i in range(len(N)):

    

        mesh = UnitIntervalMesh(N[i])
        V = FunctionSpace(mesh,'Lagrange',p)
        V2 = FunctionSpace(mesh,'Lagrange',p+3)
    
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
            
            
            E_Linf.append(max(abs(u2.vector().array()-ue.vector().array())))
            E_L1.append(assemble(abs(u-ue)*dx))
            E_L2.append(errornorm(u,ue,'L2')/sqrt(Hp_sink(p+2,K[j])))
            E.append(errornorm(u,ue,'H1')/sqrt(Hp_sink(p+1,K[j])))


        
            LS_h.append(1./N[i])
        

    

    E = array(E)
    E_L2 = array(E_L2)
    E_Linf=array(E_Linf)
    E_L1 = array(E_L1)
    LS_h = array(LS_h)
    

    valH1 = LS(zeros(len(LS_h))+1,log(LS_h),log(E))
    valL2 = LS(zeros(len(LS_h))+1,log(LS_h),log(E_L2))
    valL1 = LS(zeros(len(LS_h))+1,log(LS_h),log(E_L1))
    valLinf = LS(zeros(len(LS_h))+1,log(LS_h),log(E_Linf))

    
    return valH1,valL2,valL1,valLinf
    
    

N = [20,100,1000]
K = [10]
P=[1,2,3,4]


H1=[]
L2=[]
L1=[]
LInf=[]
for i in P:
    
    [a,b,c,d]=findRate(i,K,N)

    H1.append(a)
    L2.append(b)
    L1.append(c)
    LInf.append(d)

print
print "Least square results"
print   
for j in range(len(P)):
    print "polynomial order p=%d" % P[j], "k = ",K
    print
    print "H1: C=%e alpha=%f" % (exp(H1[j][0,0]),H1[j][1,0])
    print
    print "L2: C=%e alpha=%f" % (exp(L2[j][0,0]),L2[j][1,0])
    
    print
    print "L1: C=%e alpha=%f" % (exp(L1[j][0,0]),L1[j][1,0])
    print
    print "LInf: C=%e alpha=%f" % (exp(LInf[j][0,0]),LInf[j][1,0])
    print

    
    
"""
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
"""
