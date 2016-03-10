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



N = [10,20,30]

h_val = zeros(len(N))
E_val = zeros((5,len(N)))
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

    #P2-P0
    V3=V1
    Q3=Q2
    
    #P2-P2
    V4=V1
    Q4=FunctionSpace(mesh,"Lagrange",2)    

    #P1-P0
    V5=VectorFunctionSpace(mesh,"Lagrange",1)
    Q5=Q2
    
    S=[[V1,Q1],[V2,Q2],[V3,Q3],[V4,Q4],[V5,Q5]]

    V_E=VectorFunctionSpace(mesh,"Lagrange",5)
    Q_E=FunctionSpace(mesh,"Lagrange",5)

    for j in range(len(S)):
        W=MixedFunctionSpace([S[j][0],S[j][1]])

        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)

        f=Constant([0,0])

        ue=Expression(["x[1]*(1-x[1])","0.0"])
        pe=Expression("-2+2*x[0]")

        bc_u=DirichletBC(W.sub(0),ue,u_boundary)
        bc = [bc_u]

        a = inner(grad(u),grad(v))*dx + div(u)*q*dx + div(v)*p*dx
        L = inner(f,v)*dx

        UP=Function(W)
        solve(a==L,UP,bc)

        U,P=UP.split()
    
        #plot(U)
        #plot(P)
        
        UE =interpolate(ue,V_E)
        PE = interpolate(pe,Q_E)

        E_val[j][i] = errornorm(U,UE,'H1')+errornorm(P,PE,'L2')

print "Error:"
print E_val
print
Element=["Taylor-Hood","Crouzeix-Raviart","P2-P0"]

for i in range(3):
    A= vstack([log(h_val),ones(len(N))]).T
    
    
    LS=linalg.lstsq(A, log(E_val[i]))[0]

    
    print Element[i]
    print "con. rate: %f C=%f" %(LS[0],exp(LS[1]))
    print
