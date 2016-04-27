from dolfin import *
import time
from scipy import linalg, random
from numpy import zeros,sqrt,array,matrix
import matplotlib.pyplot as plt

def jacobi_iter(A,b,x0,tol):
    
    n= len(x0)
    x1= zeros(n)
    x = zeros(n)
    x1=x0.copy()
    x=x1.copy()
    k=0
    
    A=matrix(A.array())
    b=array(b.array())
    print 
    while sqrt(((array(A.dot(x)-b))**2).sum())/sqrt(((array(A.dot(x0)-b))**2).sum())>tol:
        
        
        for i in range(n):
            s=0
            for j in range(i):
                s = s+ x1[j]*A[i,j]
            for j in range(i+1,n):
                s = s+ x1[j]*A[i,j]
            x[i] =(b[i]- s)/A[i,i]
        
        
        x1 =x.copy()
        k=k+1
        #print sqrt(((array(A.dot(x)-b))**2).sum()),k
    print "k=%d iteretions for n=%d unknowns"%( k,n)
    return x

def solving_time(A,b):
    U = Function(V)
    t0=time.time()
    x0=random.rand(len(b.array()))
    
    x=jacobi_iter(A,b,x0,10**(-4))
    t1=time.time()
    print
    #print sqrt(sum((x-xe)**2))/len(x)
    print
    return t1-t0
T=[]
N_val = [4,8,16,32]
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

    
    t2 = solving_time(A,b)
    T.append(t2)
    #print t2

plt.plot(array(N_val),array(T))
plt.xlabel('dofs')
plt.ylabel('time in seconds')
plt.show()
