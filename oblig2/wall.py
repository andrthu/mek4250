from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log,exp
from scipy import linalg, random
import matplotlib.pylab as plt

#defining the dirichlet boudary
def u_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==0 or x[0]==1 or x[1]==1:
            return True
        else:
            return False
    else:
        return False

#defining the Neumann boudary
def p_boundary(x,on_boundary):
    if on_boundary:
        if x[1]==0:
            return True
        else:
            return False
    else:
        return False

#defining the wall part of the boundary for 7.7
def wall_stress_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==0:
            return True
        else:
            return False
    else:
        return False

#Extra boundary definition for integration, Neumann part
class N_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return p_boundary(x,on_boundary)
#Extra boundary definition for integration, wall stress part
class Wall_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return wall_stress_boundary(x,on_boundary)


#mesh resolution
N = [8,16,32,64]

#lists to store errors and convergence rates
h_val = zeros(len(N))
E_val = zeros((5,len(N)))
Wall_stress = zeros((5,len(N)))
E2 = zeros((5,len(N)))
E3= zeros((5,len(N)))
con =[]

#solving for all elements using diffrent mesh resolutions
for i in range(len(N)):

    #define the mesh
    mesh = UnitSquareMesh(N[i],N[i])
    h_val[i]=mesh.hmax()
    
    #defining different elements
    #P4-P3
    V1=VectorFunctionSpace(mesh,"Lagrange",4)
    Q1=FunctionSpace(mesh,"Lagrange",3)

    #P4-P2
    V2=VectorFunctionSpace(mesh,"Lagrange",4)
    Q2=FunctionSpace(mesh,"Lagrange",2)

    #P3-P2
    V3=VectorFunctionSpace(mesh,"Lagrange",3)
    Q3=Q2

    
    #P3-P1
    V4=V3
    Q4=FunctionSpace(mesh,"Lagrange",1)    

    #P4-P1
    V5 = V1
    Q5 = Q4
    
    # list of all element combos to solve equation for
    S=[[V1,Q1],[V2,Q2],[V3,Q3],[V4,Q4],[V5,Q5]]
    
    #spaces to interpolate exact solution onto.
    V_E=VectorFunctionSpace(mesh,"Lagrange",6)
    Q_E=FunctionSpace(mesh,"Lagrange",5)
    
    #solve for different types of elements
    for j in range(len(S)):
        
        ##create function space with spaces deined above
        W=MixedFunctionSpace([S[j][0],S[j][1]])
        
        #the normal Trial and test functions call
        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)

        #source term 
        f=Expression(["pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])","pi*pi*cos(pi*x[0])"])
        #Neumann function for variational form
        h = Expression(["-pi","-sin(2*pi*x[0])"])
        #exact velocity solution
        ue=Expression(["sin(pi*x[1])","cos(pi*x[0])"])
        
        #exact preasure solution
        pe=Expression("sin(2*pi*x[0])")
        
        #dirichlet only on part of the boundary
        bc_u=DirichletBC(W.sub(0),ue,u_boundary)
        bc = [bc_u]

        #do a lot of things to be able to integrate only on parts
        #of the boundary.
        NeumannBoundary = N_boundary()
        WallBoundary = Wall_boundary()
        boundaries = FacetFunction("size_t",mesh)
        boundaries.set_all(0)
        NeumannBoundary.mark(boundaries, 1)
        WallBoundary.mark(boundaries, 2)
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

        #the variational formula
        a = inner(grad(u),grad(v))*dx + div(u)*q*dx + div(v)*p*dx
        #notice the extra Neumann term
        L = inner(f,v)*dx + inner(h,v)*ds(1)
        
        #solve it
        UP=Function(W)
        solve(a==L,UP,bc)
        
        #split the function up to simlify finding the error
        U,P=UP.split()
    
        #plot(U)
        #plot(P)
        #interactive()
        UE =interpolate(ue,V_E)
        PE = interpolate(pe,Q_E)

        
        #defining the wall shear stress
        ws = (U-UE).dx(1)
        stress = sqrt(assemble( inner(ws,ws)*ds(2)))
        Wall_stress[j][i] = stress
        
        #find all errors
        E_val[j][i] = errornorm(ue,U,'H1')+errornorm(pe,P,'L2')
        E2[j][i]= errornorm(ue,U,'H1')
        E3[j][i]=errornorm(pe,P,'L2')

#print the errors
print "Error H1+L2:"
print E_val
print
print "Error H1"
print E2
print
print "Error L2"
print E3
print
print "Wall stress error"
print Wall_stress


#list for nicer print out
Element=["P4-P3","P4-P2","P3-P2","P3-P1","P4-P1"]
#list for plotting
expected_convergence=[4,3,3,2]

#calculate convergance rates with leas squares and print it out. 
for i in range(5):
    A= vstack([log(h_val),ones(len(N))]).T
    
    
    LS=linalg.lstsq(A, log(E_val[i]))[0]

    wall_LS=linalg.lstsq(A, log(Wall_stress[i]))[0]

    #plt.plot(log(h_val),log(E_val[i]))
    #plt.plot(log(h_val),expected_convergence[i]*log(h_val))
    #plt.show()
    print "-----------------------------------"
    print Element[i]
    print "con. rate: %f C=%f" %(LS[0],exp(LS[1]))
    print
    print "Wall stress:"
    print "con. rate: %f C=%f" %(wall_LS[0],exp(wall_LS[1]))
    print "-----------------------------------"
    print

#do some fancy plotts.
fig,ax = plt.subplots(2, 2)
for i in range(4):
    x = i/2
    y= i%2

    ax[x,y].plot(log(h_val),log(E_val[i]))
    ax[x,y].plot(log(h_val),expected_convergence[i]*log(h_val))
    ax[x,y].set_title("loglog plot for "+Element[i])
    ax[x,y].legend(["error",str(expected_convergence[i])+"h"],loc=4)
    ax[x,y].set_xlabel("h")
    ax[x,y].set_ylabel("error")
plt.show()
fig,ax = plt.subplots(2, 2)

for i in range(4):
    x = i/2
    y= i%2
    j=i
    if i==3:
        j=4
    ax[x,y].plot(log(h_val),log(Wall_stress[j]))
    #ax[x,y].plot(log(h_val),expected_convergence[i]*log(h_val))
    ax[x,y].set_title("loglog plot for "+Element[j])
    ax[x,y].legend(["error"])
    ax[x,y].set_xlabel("h")
    ax[x,y].set_ylabel("error")
plt.show()

"""
terminal>> python wall.py
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Error H1+L2:
[[  7.76762337e-05   4.54494113e-06   2.77410367e-07   1.72164886e-08]
 [  3.14124400e-03   4.45571437e-04   5.88779761e-05   7.53800432e-06]
 [  2.89622116e-03   4.18097521e-04   5.56156795e-05   7.13947628e-06]
 [  3.43304989e-02   8.24444895e-03   2.03837964e-03   5.08245315e-04]
 [  3.44711076e-02   8.26180044e-03   2.04049314e-03   5.08504813e-04]]

Error H1
[[  2.44596946e-05   1.29790898e-06   7.59505780e-08   4.64836480e-09]
 [  1.56263074e-03   2.22262656e-04   2.94067129e-05   3.76701818e-06]
 [  1.31433397e-03   1.94615723e-04   2.61359437e-05   3.36807202e-06]
 [  1.70807415e-02   4.11200938e-03   1.01794145e-03   2.53968724e-04]
 [  1.72207492e-02   4.12935105e-03   1.02005483e-03   2.54228220e-04]]

Error L2
[[  5.32165392e-05   3.24703215e-06   2.01459789e-07   1.25681238e-08]
 [  1.57861326e-03   2.23308781e-04   2.94712632e-05   3.77098613e-06]
 [  1.58188718e-03   2.23481798e-04   2.94797358e-05   3.77140426e-06]
 [  1.72497574e-02   4.13243957e-03   1.02043818e-03   2.54276592e-04]
 [  1.72503585e-02   4.13244939e-03   1.02043831e-03   2.54276593e-04]]

Wall stress error
[[  1.13537354e-05   7.10182655e-07   4.43953717e-08   2.77487425e-09]
 [  1.13537354e-05   7.10182665e-07   4.43953812e-08   2.77485127e-09]
 [  4.29061351e-04   5.37004852e-05   6.71468165e-06   8.39401549e-07]
 [  4.29061351e-04   5.37004852e-05   6.71468165e-06   8.39401550e-07]
 [  1.13537354e-05   7.10182662e-07   4.43953789e-08   2.77485354e-09]]
-----------------------------------
P4-P3
con. rate: 4.045257 C=0.084663

Wall stress:
con. rate: 3.999508 C=0.011619
-----------------------------------

-----------------------------------
P4-P2
con. rate: 2.902867 C=0.493978

Wall stress:
con. rate: 3.999512 C=0.011619
-----------------------------------

-----------------------------------
P3-P2
con. rate: 2.890269 C=0.447429

Wall stress:
con. rate: 2.999237 C=0.077591
-----------------------------------

-----------------------------------
P3-P1
con. rate: 2.024947 C=1.135217

Wall stress:
con. rate: 2.999237 C=0.077591
-----------------------------------

-----------------------------------
P4-P1
con. rate: 2.026649 C=1.142794

Wall stress:
con. rate: 3.999511 C=0.011619
-----------------------------------

"""

