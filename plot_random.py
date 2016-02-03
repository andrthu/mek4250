from dolfin import *
from numpy import matrix, sqrt, diagflat,zeros
from scipy import linalg, random

def boundary(x,on_boundary): return on_boundary

N=100


mesh = UnitIntervalMesh(N)
V = FunctionSpace(mesh,"Lagrange",1)
u = TrialFunction(V)
v = TestFunction(V)
bc=DirichletBC(V,Constant(0),boundary)



a = random.rand(N+1)
a[0]=0
a[-1]=0
f = Function(V)
f.vector()[:]=a

plot(f)
interactive()
