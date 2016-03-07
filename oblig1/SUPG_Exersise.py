from dolfin import *
from numpy import pi,matrix, sqrt, diagflat,zeros,vstack,ones,log,array
#from scipy.special import factorial as fac
from math import factorial as fac
from scipy import linalg





def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False

con = []
errorL2 = []
errorH1 = []

for my in [1,0.001,0.000001]:

    L2 = []
    H1 = []
    h_val = []

    for h in [8,16,32,64]:

                
        mesh = UnitSquareMesh(h,h)

        beta = mesh.hmax()
        
        V = FunctionSpace(mesh,'Lagrange',1)
        V2 = FunctionSpace(mesh,'Lagrange',1+3)
        
        
        f = Constant(0)
        if my==1:
            ue = Expression('(1-exp(x[0]))/(1-exp(1))')
        else:
            ue = Expression('exp((x[0]-1)/%e)'%my)
        g = Expression('x[0]')
            
        
        u = TrialFunction(V)
        v = TestFunction(V)

        MY = Constant(my)
        Beta = Constant(beta)
        
        a = (MY*inner(grad(u),grad(v))+v*u.dx(0)+Beta*u.dx(0)*v.dx(0))*dx
        L = (f*v+Beta*f*v.dx(0))*dx 

                
                
        bc = DirichletBC(V,g,Dirichlet_boundary)

        U = Function(V)

        solve(a==L,U,bc)
        Ue = interpolate(ue,V2)
        """
        print
        print errornorm(U,Ue,'H1')
        print errornorm(U,Ue)
        print my,h
        print assemble(Ue**2*dx)
        """
        L2.append(errornorm(U,Ue))
        H1.append(errornorm(U,Ue,'H1'))
        h_val.append( mesh.hmax())
        """
        if my == 0.001 and h==64:
            
            print errornorm(U,Ue)
            print assemble((U-Ue)**2*dx)
        """
        plot(Ue)
        interactive()
    errorL2.append(L2)
    errorH1.append(H1)
    Q = vstack([log(array(h_val)),ones(len(h_val))]).T
    con.append(linalg.lstsq(Q, log(array(L2)))[0])
    con.append(linalg.lstsq(Q, log(array(H1)))[0])
for i in range(3):
    print "my=%e" % (0.001**i)
    print "L2 Error: ", errorL2[i]
    print
    print "H1 Error: ", errorH1[i]
    print 
    print "L2 convergence: ", con[2*i]
    print
    print "H1 convergance: ", con[2*i+1]
