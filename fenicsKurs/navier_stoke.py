from dolfin import *

mesh = Mesh("dolfin_channel .xml")
#plot(mesh)
#interactive()
n=FacetNormal(mesh)

V = VectorFunctionSpace(mesh,"Lagrange",2)
Q = FunctionSpace(mesh,"Lagrange",1)
W = V*Q

mu=0.001002
rho = 1000
k= 0.0005
T=0.1
def inflow_dom(x):
    return x[0]<DOLFIN_EPS and x[1]>DOLFIN_EPS and x[1]-DOLFIN_EPS

def outflow_dom(x)
    return x[0]>1-DOLFIN_EPS and x[1]>DOLFIN_EPS and x[1]-DOLFIN_EPS
def noslip(x):
    return
def sigma(u,p):
    return 2.0*mu*sym(grad(u))-p*Identity(len(u))

u = TrialFunction(V)
p= TrialFunction(Q)
v = TestFunction(V)
q= TestFunction(Q)

u0=Function(V)
p0=Function(Q)
us = Function(V)


Rho=Constant(rho)
Mu=Constant(mu)
dt=Constant(k)
"""
F1 = Rho*inner(u-u0,v)*dx + dt*Rho*inner(dot(u0,nabla_grad(u0)),v)*dx + inner(sigma(0.5*(u+u0),p0),sym(grad(v)))*dx - Mu*inner(dot(grad(0.5*(u+u0)),n),v)*ds +inner(p0,v)*ds
"""
f1 = Rho*inner(u-u0,v)*dx + dt*Rho*inner(dot(u0,nabla_grad(u0)),v)*dx
f2 = dt*inner(sigma(0.5*(u+u0),p0),sym(grad(v)))*dx
f3 = - dt*Mu*inner(dot(grad(0.5*(u+u0)),n),v)*ds
f4 = dt*inner(p0*n,v)*ds
F1 = f1+f2+f3+f4


a1 = lhs(F1)
L1 = rhs(F1)

a2 = dt*inner(grad(p),grad(q))*dx
L2 = dt*inner(grad(p0),grad(q))*dx-Rho*div(us)*q*dx
A2 = assemble(a2)

t = k
bc1 = DirichletBC(V,Constant((0.0,0.0)),"x[0]>DOLFIN_EPS && x[0]<1.0-DOLFIN_EPS && on_boundary")
bc2=DirichletBC(Q,Constant(1000.0),"x[0]<DOLFIN_EPS")

u1=Function(V)
P=Function(Q)

a3 = Rho*inner(u,v)*dx
L3 = Rho*inner(us,v)*dx-dt*inner(grad(P-p0),v)*dx

u0 = interpolate(Constant((0.0,0.0)),V)
p0=interpolate(Constant(0.0),Q)
while t<=T:
    solve(a1==L1,us,bc1)

    solve(a2==L2,P,bc2)

    solve(a3==L3,u1, bc1,solver_parameters={"linear_solver": "lu"})
    
    p0.assign(P)
    u0.assign(u1)
    t=t+k

plot(u1)
interactive()
print assemble(u1[1]*dx)/assemble(1.0*dx(mesh))
