from dolfin import *
#from dolfin_adjoint import *


def boundary(x,on_boundary):
    if on_boundary:
        return x[0]==0

mesh = UnitIntervalMesh(100)

V = FunctionSpace(mesh,'CG',1)

u = TrialFunction(V)
v = TestFunction(V)

y = interpolate(Constant(0.0),V)

a = (u.dx(0)*v - v*u)*dx
L = y*v*dx

ic = 1.0
bc = DirichletBC(V, Constant(ic),boundary)

U = Function(V)

solve(a==L,U,bc)

#plot(u)
#interactive()
