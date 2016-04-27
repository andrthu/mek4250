from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros,vstack,ones,log,exp
from scipy import linalg, random

def u_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==0 or x[1]==1 or x[1]==0:
            return True
        else:
            return False
    else:
        return False

def p_boundary(x,on_boundary):
    if on_boundary:
        if x[0]==1:
            return True
        else:
            return False
    else:
        return False



N = [4,8,16]

h_val = zeros(len(N))
E_val = zeros((5,len(N)))
E2 = zeros((5,len(N)))
E3= zeros((5,len(N)))
K = zeros((5,len(N)))
con =[]

for i in range(len(N)):

    

    mesh = UnitSquareMesh(N[i],N[i])
    h_val[i]=mesh.hmax()
    
    #Taylor-hood=P2-P1
    V1=VectorFunctionSpace(mesh,"Lagrange",2)
    Q1=FunctionSpace(mesh,"Lagrange",1)

    #Cruz-rav
    V2=VectorFunctionSpace(mesh,"CR",1)
    Q2=FunctionSpace(mesh,"DG",0)

    #mini
    """
    p1 = VectorElement("Lagrange", mesh.ufl_cell(),1)
    b  = VectorElement("Bubble",mesh.ufl_cell(), 3)
    Q = FiniteElement("Lagrange", mesh.ufl_cell(),1)
    W3=MixedFunctionSpace(mesh,(p1+b)*Q)
    
    P1 = VectorFunctionSpace(mesh, "Lagrange", 1)
    B  = VectorFunctionSpace(mesh, "Bubble", 3)
    Q  = FunctionSpace(mesh, "CG",  1)
    V = P1 + B
    W3 = V*Q
    """
    #V4 = P1+B
    #Q4 = Q1
    
    
    S=[[V1,Q1],[V2,Q2]]

    

    for j in range(len(S)):
        
        if j==2:
            W=S[j]
        else:
            W=MixedFunctionSpace([S[j][0],S[j][1]])

        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)

        f=Constant([0,0])
        #f=Expression(["4*pi*pi*sin(2*pi*x[1])","0.0"])
        #ue=Expression(["sin(2*pi*x[1])+x[1]*(1-x[1])","0.0"])
        ue=Expression(["x[1]*(1-x[1])","0.0"])
        #pe=Expression("-2+2*x[0]")

        bc_u=DirichletBC(W.sub(0),ue,u_boundary)
        bc = [bc_u]
        
        A,_ = assemble_system((inner(grad(u),grad(v))+div(u)*q+div(v)*p)*dx,inner(f,v)*dx,bc)


        l,v = linalg.eigh(matrix(A.array()))


        l1 =float(max(abs(l)))
        l2 = min(abs(l))
        
        K[j][i] = l1/l2
        print l1,l2,l1/l2

        
Element=["Taylor-Hood","Crouzeix-Raviart","P2-P0","mini"]        
for i in range(2):
    R= vstack([log(h_val),ones(len(N))]).T
    
    
    LS=linalg.lstsq(R, log(K[i]))[0]

    print '---------------------------'
    print Element[i]
    print
    print 'con rate: ',LS[0]
    print
    print 'condition numbers: ' , K[i]
    print '---------------------------'
    print 
    print

print K[0]/K[1]
