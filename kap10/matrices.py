from dolfin import *
import numpy as np

mesh = UnitSquareMesh(5,5)

V = VectorFunctionSpace(mesh,'Lagrange',1)
Q=FunctionSpace(mesh,"Lagrange",1)
W=MixedFunctionSpace([V,Q])

u,p = TrialFunctions(W)
v,q = TestFunctions(W)

A1 = inner(grad(u),grad(v))*dx
A2 = inner(grad(p),grad(q))*dx
A3 = div(v)*p*dx

g = Expression(['0.0','0.0'])
b1 = dot(g,v)*dx
b2 = Constant(0)*q*dx

bc1 = DirichletBC(W.sub(0),g,"on_boundary") 
bc2 = DirichletBC(W.sub(1),Constant(0),"on_boundary")

AA,_ = assemble_system(A3,b1,bc1)
AA2,_= assemble_system(A2,b2,bc2)

print AA.array()
print
print AA2.array()
