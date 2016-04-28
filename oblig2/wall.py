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
E_val = zeros((4,len(N)))
E2 = zeros((4,len(N)))
E3= zeros((4,len(N)))
con =[]

for i in range(len(N)):

    mesh = UnitSquareMesh(N[i],N[i])
    h_val[i]=mesh.hmax()
    
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

    
    
    S=[[V1,Q1],[V2,Q2],[V3,Q3],[V4,Q4]]

    V_E=VectorFunctionSpace(mesh,"Lagrange",6)
    Q_E=FunctionSpace(mesh,"Lagrange",5)

    for j in range(len(S)):
        
        W=MixedFunctionSpace([S[j][0],S[j][1]])

        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)

        
        f=Expression(["pi*pi*sin(pi*x[1])-2*pi*cos(2*pi*x[0])","pi*pi*cos(pi*x[0])"])
        ue=Expression(["sin(pi*x[1])","cos(pi*x[0])"])
        
        pe=Expression("sin(2*pi*x[0])")

        bc_u=DirichletBC(W.sub(0),ue,u_boundary)
        bc = [bc_u]

        a = inner(grad(u),grad(v))*dx + div(u)*q*dx + div(v)*p*dx
        L = inner(f,v)*dx

        UP=Function(W)
        solve(a==L,UP,bc)

        U,P=UP.split()
    
        #plot(U)
        #plot(P)
        #interactive()
        UE =interpolate(ue,V_E)
        PE = interpolate(pe,Q_E)

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
Element=["P4-P3","P4-P2","P3-P2","P3-P1"]

for i in range(4):
    A= vstack([log(h_val),ones(len(N))]).T
    
    
    LS=linalg.lstsq(A, log(E_val[i]))[0]

    
    print Element[i]
    print "con. rate: %f C=%f" %(LS[0],exp(LS[1]))
    print
