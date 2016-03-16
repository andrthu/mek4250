from dolfin import *
from numpy import pi,matrix, sqrt, diagflat,zeros,vstack,ones,log,array,exp
#from scipy.special import factorial as fac
from math import factorial as fac
from scipy import linalg




#function to define the Dirichlet boundary
def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False

#lists to store errors and convergence
con = []
errorL2 = []
errorH1 = []

#Solve the equaton for different mu
for my in [1,0.1,0.01,0.001,0.0001]:
    
    #more error storing
    L2 = []
    H1 = []
    h_val = []
    
    #size of mesh
    for h in [8,16,32,64]:

        #define mesh and functionspaces
        mesh = UnitSquareMesh(h,h)        
        V = FunctionSpace(mesh,'Lagrange',1)
        V2 = FunctionSpace(mesh,'Lagrange',1+3)
        
        
        #define right hand side function
        f = Constant(0)

        #define exact solution. do trick to properly get what I want when mu is very small
        if my>0.001:
            ue = Expression('(1-exp(x[0]/%e))/(1-exp(1/%e))'%(my,my))
        else:
            ue = Expression('exp((x[0]-1)/%e)'%my)
            
        #g is function on the boundary u(0,y)=0 u(1,y)=1
        g = Expression('x[0]')
            
        
        #define variational formulatin
        u = TrialFunction(V)
        v = TestFunction(V)

        MY = Constant(my)
        
        a = MY*inner(grad(u),grad(v))*dx +v*u.dx(0)*dx
        L = f*v*dx 

                
        #define boudary conditions
        bc = DirichletBC(V,g,Dirichlet_boundary)
        
        #solve the equation
        U = Function(V)
        solve(a==L,U,bc)

        #find the error in L2 and H1 norm
        Ue = interpolate(ue,V2)
       
        L2.append(errornorm(U,Ue))
        H1.append(errornorm(U,Ue,'H1'))
        h_val.append( mesh.hmax())
        
        if my==0.001 and h==64:
            plot(U)
            interactive()
    #calculate the convergence rate for each mu
    errorL2.append(L2)
    errorH1.append(H1)
    Q = vstack([log(array(h_val)),ones(len(h_val))]).T
    con.append(linalg.lstsq(Q, log(array(L2)))[0])
    con.append(linalg.lstsq(Q, log(array(H1)))[0])


for i in range(5):
    print "---------------my=%e---------------------" % (0.1**i)
    print "L2 Error: ", errorL2[i]
    print
    print "H1 Error: ", errorH1[i]
    print 
    print "L2 convergence=%f , constant=%f " %(con[2*i][0],exp(con[2*i][1]))
    print
    print "H1 convergance=%f , constant=%f " %(con[2*i+1][0],exp(con[2*i+1][1]))
    print "------------------------------------------"
    print

"""
terminal> python Exercise2.py

---------------my=1.000000e+00---------------------
L2 Error:  [0.0014024911398510243, 0.00035075775846229513, 8.7698381572637e-05, 2.1925164775258797e-05]

H1 Error:  [0.037522413566527565, 0.01876559984230818, 0.009383378966941046, 0.004691763093163726]

L2 convergence=1.999763 , constant=0.044866 

H1 convergance=0.999856 , constant=0.212219 
------------------------------------------

---------------my=1.000000e-01---------------------
L2 Error:  [0.023747305281734955, 0.006176861546736365, 0.0015613252209452253, 0.00039147123761489256]

H1 Error:  [0.7692373698326008, 0.39838931594977284, 0.2010768389408248, 0.10078145954745421]

L2 convergence=1.975224 , constant=0.735446 

H1 convergance=0.978303 , constant=4.229330 
------------------------------------------

---------------my=1.000000e-02---------------------
L2 Error:  [0.2389653886603793, 0.10398976824872899, 0.03814192062101558, 0.011254531035966306]

H1 Error:  [7.7969983422840405, 7.008644352418786, 5.086479716611928, 2.982329423874184]

L2 convergence=1.467166 , constant=3.339322 

H1 convergance=0.462191 , constant=19.327090 
------------------------------------------

---------------my=1.000000e-03---------------------
L2 Error:  [1.4205871191242874, 0.45478843938509783, 0.19169796112898158, 0.0941625964337223]

H1 Error:  [36.08584314601011, 25.48364599592582, 23.46005066643907, 24.24642271735071]

L2 convergence=1.299193 , constant=12.052507 

H1 convergance=0.184035 , constant=44.796474 
------------------------------------------

---------------my=1.000000e-04---------------------
L2 Error:  [13.576604419546138, 3.89939174630185, 0.9717085725730065, 0.3351154844364964]

H1 Error:  [313.9722174102841, 175.99443692797345, 99.33859332033042, 75.67413118513336]

L2 convergence=1.802562 , constant=301.737979 

H1 convergance=0.698340 , constant=989.689148 
------------------------------------------


"""
