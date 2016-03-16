


from dolfin import *
from numpy import pi,matrix, sqrt, diagflat,zeros,vstack,ones,log,array
from scipy import linalg


#Defining the Dirichlet boundary
def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False

#lists for storing errors and stuff
con = []
errorL2 = []
errorH1 = []


#solve equation for diffrent mu
for my in [1,0.1,0.01,0.001,0.0001]:

    #more lists for storing
    L2 = []
    H1 = []
    h_val = []

    #size of mesh
    for h in [8,16,32,64]:

        #define mesh and 
        mesh = UnitSquareMesh(h,h)
        beta = mesh.hmax()     
        V = FunctionSpace(mesh,'Lagrange',1)
        V2 = FunctionSpace(mesh,'Lagrange',1+3)
        
        
        #right hand side in equation
        f = Constant(0)

        #exact solution + trick
        if my==1:
            ue = Expression('(1-exp(x[0]))/(1-exp(1))')
        else:
            ue = Expression('exp((x[0]-1)/%e)'%my)

        #boundary function u(0,y)=0 u(1,y)=1
        g = Expression('x[0]')
            
        
        #setting up variational formulation
        u = TrialFunction(V)
        v = TestFunction(V)

        MY = Constant(my)
        Beta = Constant(beta)
        
        a = (MY*inner(grad(u),grad(v))+v*u.dx(0)+Beta*u.dx(0)*v.dx(0))*dx
        L = (f*v+Beta*f*v.dx(0))*dx 

                
        #boundary
        bc = DirichletBC(V,g,Dirichlet_boundary)
        
        #solve system
        U = Function(V)
        solve(a==L,U,bc)
        
        #calculate the error etc
        Ue = interpolate(ue,V2)
        
        
        L2.append(errornorm(U,Ue))
        H1.append(errornorm(U,Ue,'H1'))
        h_val.append( mesh.hmax())
        if my==0.001 and h==64:
            plot(U)
            interactive()

    errorL2.append(L2)
    errorH1.append(H1)
    #find convergence rate for each mu using least square.

    Q = vstack([log(array(h_val)),ones(len(h_val))]).T
    con.append(linalg.lstsq(Q, log(array(L2)))[0])
    con.append(linalg.lstsq(Q, log(array(H1)))[0])
    
for i in range(5):
    print "-----------------my=%f--------------------" % (0.1**i)
    print "L2 Error: ", errorL2[i]
    print
    print "H1 Error: ", errorH1[i]
    print
    print "L2 convergence=%f, Constant=%f "% (con[2*i][0],exp(con[2*i][1]))
    print
    print "H1 convergance%f, Constant=%f "%( con[2*i+1][0], exp(con[2*i+1][1]))
    print "-----------------------------------------------"
    print
"""
terminal> python SUPG_Exersise.py
-----------------my=1.000000--------------------
L2 Error:  [0.01416838752283591, 0.0073698629559538304, 0.003766948027708709, 0.0019055030318557316]

H1 Error:  [0.058124707898786016, 0.030553163128422527, 0.01569121125708018, 0.007955266894620535]

L2 convergence=0.965154, Constant=0.076007 

H1 convergance0.956887, Constant=0.308063 
-----------------------------------------------

-----------------my=0.100000--------------------
L2 Error:  [0.19247496357120064, 0.11447613838596869, 0.06286471681033422, 0.033037850050471426]

H1 Error:  [1.291133318726959, 0.8952364831423733, 0.5543188017552529, 0.31486132432479447]

L2 convergence=0.849216, Constant=0.866334 

H1 convergance0.679910, Constant=4.414590 
-----------------------------------------------

-----------------my=0.010000--------------------
L2 Error:  [0.28346579869290073, 0.19326250254022131, 0.12513237445344977, 0.07684858033650653]

H1 Error:  [5.62404968238124, 6.118072592987696, 5.569677858352143, 4.491879580564998]

L2 convergence=0.627636, Constant=0.863301 

H1 convergance0.110835, Constant=7.365285 
-----------------------------------------------

-----------------my=0.001000--------------------
L2 Error:  [0.28120110865768316, 0.200020969882777, 0.14226112534474597, 0.10099831438927163]

H1 Error:  [6.016784489921939, 8.508281095933974, 12.03054605209179, 16.50452716235202]

L2 convergence=0.492342, Constant=0.660256 

H1 convergance-0.486715, Constant=2.604413 
-----------------------------------------------

-----------------my=0.000100--------------------
L2 Error:  [0.28047072407456436, 0.1989474319782039, 0.14076212733913698, 0.0996538431414621]

H1 Error:  [6.015360380217439, 8.50450651604375, 12.027273462389585, 17.011364110866857]

L2 convergence=0.497769, Constant=0.664928 

H1 convergance-0.499934, Constant=2.529147 
-----------------------------------------------



"""
