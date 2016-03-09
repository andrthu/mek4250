from dolfin import *
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
    
mesh = UnitSquareMesh(100,100)
V=VectorFunctionSpace(mesh,"Lagrange",1)
Q=FunctionSpace(mesh,"Lagrange",1)
#Q=FunctionSpace(mesh,"DG",0)
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
L = inner(f,v)*dx

UP=Function(W)
solve(a==L,UP,bc)

U,P=UP.split()

plot(U)
plot(P)

UE =interpolate(ue,V)
PE = interpolate(pe,Q)
plot(UE)
plot(PE)

interactive()
