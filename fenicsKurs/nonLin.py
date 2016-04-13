
from  __future__  import  print_function
from  fenics  import *
# Use -02  optimization
parameters["form_compiler"]["cpp_optimize"] = True
# Define  mesh  and  geometry
mesh = Mesh("dolfin-2.xml")
n = FacetNormal(mesh)
# Define  Taylor --Hood  function  space W
V = VectorFunctionSpace(mesh , "Lagrange" , 2)
Q = FunctionSpace(mesh , "Lagrange", 1 )
P2 = VectorElement("Lagrange", triangle , 2)
P1 = FiniteElement("Lagrange", triangle , 1)
TH = MixedElement ([P2, P1])
W = V*Q
# Define  Function  and  TestFunction(s)

w = Function(W)
(u, p) = split(w)
(v, q) = TestFunctions(W)


 
# Define  viscosity  and bcsn
#nu = Expression("0.2*(1+pow(x[1],2))", degree=2)
#nu=Constant(0.2)
no=4
def viscocity(u):
    return 0.5*(inner(nabla_grad(u),nabla_grad(u))**(1./(2*(no-1))))
                
nu=Constant(0.2)
p0 = Expression("1.0-x[0]", degree=1)
bcs = DirichletBC(W.sub(0), (0.0, 0.0),"on_boundary  && !(near(x[0], 0.0) || near(x[0], 1.0))")

# Define  variational  form
epsilon = sym(grad(u))
F = (2*nu*inner(epsilon , grad(v)) - div(u)*q - div(v)*p)*dx + p0*dot(v,n)*ds
    
# Solve  problem
dF = derivative(F, w)
pde = NonlinearVariationalProblem(F, w, bcs , dF)
solver = NonlinearVariationalSolver(pde)
solver.solve()



nu = viscocity(u)

F = (2*nu*inner(epsilon , grad(v)) - div(u)*q - div(v)*p)*dx + p0*dot(v,n)*ds
    
# Solve  problem
dF = derivative(F, w)
pde = NonlinearVariationalProblem(F, w, bcs , dF)
solver = NonlinearVariationalSolver(pde)
solver.solve()

# Plot  solutions
plot(u, title="Velocity", interactive=True)
plot(p, title="Pressure", interactive=True)
