from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log,exp
from scipy import linalg, random
import matplotlib.pylab as plt

#defining the boudary
def u_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==0 or x[0]==1 or x[1]==1:
            return True
        else:
            return False
    else:
        return False

def p_boundary(x,on_boundary):
    if on_boundary:
        if x[1]==0:
            return True
        else:
            return False
    else:
        return False

def wall_stress_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==0:
            return True
        else:
            return False
    else:
        return False

class N_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return p_boundary(x,on_boundary)

class Wall_boundary(SubDomain):
    def inside(self, x, on_boundary):
        return wall_stress_boundary(x,on_boundary)



N = [8,16,32,64]

h_val = zeros(len(N))
E_val = zeros((5,len(N)))
Wall_stress = zeros((5,len(N)))
E2 = zeros((5,len(N)))
E3= zeros((5,len(N)))
con =[]

#solving for all elements using diffrent mesh resolutions
for i in range(len(N)):

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
    
    S=[[V1,Q1],[V2,Q2],[V3,Q3],[V4,Q4],[V5,Q5]]

    V_E=VectorFunctionSpace(mesh,"Lagrange",6)
    Q_E=FunctionSpace(mesh,"Lagrange",5)
    
    #solve for different types of elements
    for j in range(len(S)):
        
        W=MixedFunctionSpace([S[j][0],S[j][1]])

        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)

        #source term 
        f=Expression(["pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])","pi*pi*cos(pi*x[0])"])
        h = Expression(["-pi","-sin(2*pi*x[0])"])
        #exact velocity solution
        ue=Expression(["sin(pi*x[1])","cos(pi*x[0])"])
        
        #exact preasure solution
        pe=Expression("sin(2*pi*x[0])")
        
        #dirichlet only on part of the boundary
        bc_u=DirichletBC(W.sub(0),ue,u_boundary)
        #bc_p=DirichletBC(W.sub(1),pe,"on_boundary")
        bc = [bc_u]
        
        NeumannBoundary = N_boundary()
        WallBoundary = Wall_boundary()
        boundaries = FacetFunction("size_t",mesh)
        boundaries.set_all(0)
        NeumannBoundary.mark(boundaries, 1)
        WallBoundary.mark(boundaries, 2)
        ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

        #the variational formula
        a = inner(grad(u),grad(v))*dx + div(u)*q*dx + div(v)*p*dx  
        L = inner(f,v)*dx + inner(h,v)*ds(1)
        
        #solve it
        UP=Function(W)
        solve(a==L,UP,bc)

        U,P=UP.split()
    
        #plot(U)
        #plot(P)
        #interactive()
        UE =interpolate(ue,V_E)
        PE = interpolate(pe,Q_E)
        
        t = Expression(["0","1"])
        
        ws = (U-UE).dx(1)
        stress = sqrt(assemble( inner(ws,ws)*ds(2)))
        Wall_stress[j][i] = stress
        
        E_val[j][i] = errornorm(ue,U,'H1')+errornorm(pe,P,'L2')
        E2[j][i]= errornorm(ue,U,'H1')
        E3[j][i]=errornorm(pe,P,'L2')
        
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
Element=["P4-P3","P4-P2","P3-P2","P3-P1","P4-P1"]

expected_convergence=[4,3,3,2]
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

