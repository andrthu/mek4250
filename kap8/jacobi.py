from dolfin import *
import time
from scipy import linalg, random
from numpy import zeros,sqrt,array,matrix
import matplotlib.pyplot as plt
def jacobi_iter(A,b,x0,xe,tol):
    
    n= len(x0)
    x1= zeros(n)
    x = zeros(n)
    x1=x0
    x=x1
    k=0
    print len(A.array())
    A=matrix(A.array())
    b=array(b.array())

    while sqrt(sum((x-xe)**2))/sqrt(sum((x0-xe)**2))>tol:
        print len(x1[1:]),len(A[0][1:])
        x[0]=(b[0]- sum(x1[1:]*A[0,1:]))/A[0,0]
        #for i in range(1,n-1):
         #   x[i] =(b[i]- sum(x1[:i]*A[i,:i]) - sum(x1[i+1:]*A[i,i+1:]))/A[i,i]
        #x[-1] =(b[-1]- sum(x1[:-1]*A[-1,:-1]))/A[-1,-1]
        
        #x1 =x
        #k=k+1
    print k
    return x

def solving_time(A,b,ue):
    U = Function(V)
    t0=time.time()
    xe = ue.vector().array()
    x0=random.rand(len(xe))
    
    x=jacobi_iter(A,b,x0,xe,10**(-4))
    t1=time.time()
    print
    #print sqrt(sum((x-xe)**2))/len(x)
    print
    return t1-t0
T=[]
N_val = [32,64,128,256,512,1024,2048]
for N in N_val:

    mesh = UnitIntervalMesh(N)

    V = FunctionSpace(mesh,'Lagrange',1)
    u = TrialFunction(V)
    v = TestFunction(V)

    f = Expression('pi*pi*sin(pi*x[0])')
    Ue = Expression('sin(pi*x[0])')
    ue = interpolate(Ue,V)
    a = inner(grad(u),grad(v))*dx
    L = f*v*dx
    
    bc = DirichletBC(V,Constant(0),"on_boundary")
    A,b = assemble_system(a,L,bc)

    t2 = solving_time(A,b,ue)
    T.append(t2)
    #print t2

plt.plot(array(N_val),array(T))
plt.show()
