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


N = [10,20,40]

h_val = zeros(len(N))
E_val = zeros(len(N))
con =[]
for i in range(len(N)):

    
    mesh = UnitSquareMesh(N[i],N[i])
    V=VectorFunctionSpace(mesh,"Lagrange",1)
    V2=VectorFunctionSpace(mesh,"Lagrange",4)
    Q=FunctionSpace(mesh,"Lagrange",1)
    Q2=FunctionSpace(mesh,"Lagrange",4)
    W=MixedFunctionSpace([V,Q])
    
    

    
    u,p=TrialFunctions(W)
    v,q=TestFunctions(W)

    f=Constant([0,0])

    ue=Expression(["x[1]*(1-x[1])","0.0"])
    pe=Expression("-2+2*x[0]")

    bc_u=DirichletBC(W.sub(0),ue,u_boundary)
    bc = [bc_u]


    epsilon=0.01*mesh.hmax()
    eps=Constant(epsilon)

    a = (inner(grad(u),grad(v))+div(u)*q+div(v)*p-eps*inner(grad(p),grad(q)))*dx
    #a = (inner(grad(u),grad(v))+div(u)*q+div(v)*p-eps*p*q)*dx
    L = inner(f,v)*dx

    UP=Function(W)
    solve(a==L,UP,bc)

    U,P=UP.split()

    

    UE =interpolate(ue,V2)
    PE = interpolate(pe,Q2)
    h_val[i]=mesh.hmax()
    E_val[i] = errornorm(U,UE,'H1')+errornorm(P,PE,'L2')

    
    plot(UE)
    plot(PE)

    interactive()


A= vstack([log(h_val),ones(len(N))]).T
LS=linalg.lstsq(A, log(E_val))[0]
print "h values:"
print h_val
print
print "Error:"
print E_val
print
print "con_rate=%f C=%f" % (LS[0],exp(LS[1]))
print
