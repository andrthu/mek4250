from dolfin import *

mesh = Mesh("lego_beam.xml")


V=VectorFunctionSpace(mesh,"Lagrange",1)
"""
left = CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
"""
class  MyBoundary1(SubDomain):
    def  inside(self , x, on_boundary):
        #return  on_boundary  and near(x[0],10.0)
        return x[0]> 10.0*0.008 - 0.1*0.001 - DOLFIN_EPS

class  MyBoundary2(SubDomain):
    def  inside(self , x, on_boundary):
        #return  on_boundary  and near(x[0],0.0)
        return x[0]<0.1*0.001 + DOLFIN_EPS

my_boundary_1 = MyBoundary1()
my_boundary_2 = MyBoundary2()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
my_boundary_1.mark(boundaries,1)
my_boundary_2.mark(boundaries,2)
ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

I_ = Expression(("x[0]","x[1]","x[2]"))
"""
I = interpolate(I_,V)
u=TrialFunction(V)
F1=grad(u+I)
E = variable(0.5*F1.T*F1-grad(I))

l = Constant(0.0105)
m = Constant(0.00230)
B=Expression(["0.0","0.0","-9.81"])
T=Constant(["0.0","0.0","5000"])

W = 0.5*l*(tr(E))**2+m*tr(E*E)
S=diff(W,E)
P=F1*S
"""
#define func
du = TrialFunction(V)
v=TestFunction(V)
u= Function(V)
B=Constant(("0.0","0.0","-14224.5"))
T=Constant(("0.0","0.0","-5000"))

#konst.
l = Constant(0.0105*1e9)
m = Constant(0.00230*1e9)

#kinematics
d = u.geometric_dimension()
I=Identity(d)
F1 = I +grad(u)
C=F1.T*F1
E = variable(0.5*(C-I))
W = 0.5*l*(tr(E))**2 + m*tr(E*E)
S=diff(W,E)
P=F1*S


F= inner(P,grad(v))*dx - inner(B,v)*dx - inner(T,v)*ds(1)
J = derivative(F,u,du)
bc = DirichletBC(V, Constant(("0.0","0.0","0.0")), "x[0]<0.1*0.001 + DOLFIN_EPS" )
#on_boundary && !(near(x[0],10.0) || near(x[0], 0.0))
U=Function(V)
#solve(F==0,u,bc)
solve(F==0,u,bc,J=J)

plot(u)
interactive()

r=u.sub(2,deepcopy=True).vector().array()
print assemble(u[2]*dx)/assemble(1.0*dx(mesh))
