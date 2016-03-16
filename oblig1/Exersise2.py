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

for my in [1,0.1,0.01,0.001,0.0001]:

    L2 = []
    H1 = []
    h_val = []

    for h in [8,16,32,64]:

                
        mesh = UnitSquareMesh(h,h)
        
        V = FunctionSpace(mesh,'Lagrange',1)
        V2 = FunctionSpace(mesh,'Lagrange',1+3)
        
        
        f = Constant(0)
        if my>0.001:
            ue = Expression('(1-exp(x[0]/%e))/(1-exp(1/%e))'%(my,my))
        else:
            ue = Expression('exp((x[0]-1)/%e)'%my)
        g = Expression('x[0]')
            
        
        u = TrialFunction(V)
        v = TestFunction(V)

        MY = Constant(my)
        
        a = MY*inner(grad(u),grad(v))*dx +v*u.dx(0)*dx
        L = f*v*dx 

                
                
        bc = DirichletBC(V,g,Dirichlet_boundary)

        U = Function(V)

        solve(a==L,U,bc)
        Ue = interpolate(ue,V2)
        """
        print
        print errornorm(U,Ue,'H1')
        print errornorm(U,Ue)
        print my,h
        print
        """
        L2.append(errornorm(U,Ue))
        H1.append(errornorm(U,Ue,'H1'))
        h_val.append( mesh.hmax())
        """
        if my == 0.001 and h==64:
            
            print errornorm(U,Ue)
            print assemble((U-Ue)**2*dx)
            plot(U)
            interactive()
        """

    errorL2.append(L2)
    errorH1.append(H1)
    Q = vstack([log(array(h_val)),ones(len(h_val))]).T
    con.append(linalg.lstsq(Q, log(array(L2)))[0])
    con.append(linalg.lstsq(Q, log(array(H1)))[0])


for i in range(5):
    print "my=%e" % (0.1**i)
    print "L2 Error: ", errorL2[i]
    print
    print "H1 Error: ", errorH1[i]
    print 
    print "L2 convergence: ", con[2*i]
    print
    print "H1 convergance: ", con[2*i+1]
    print
