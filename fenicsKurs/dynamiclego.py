from dolfin import *

mesh = Mesh("lego_beam.xml")


V=VectorFunctionSpace(mesh,"Lagrange",1)
VV=V*V



class  MyBoundary1(SubDomain):
    def  inside(self , x, on_boundary):
        return  x[0]< 0.001 + DOLFIN_EPS


class  MyBoundary2(SubDomain):
    def  inside(self , x, on_boundary):
        return  x[0]> 10.0*0.008 -0.001 -DOLFIN_EPS

my_boundary_1 = MyBoundary1()
my_boundary_2 = MyBoundary2()

boundaries = FacetFunction("size_t",mesh)
boundaries.set_all(0)
my_boundary_1.mark(boundaries,1)
my_boundary_2.mark(boundaries,2)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

#def functions
up0 = Function(VV)
up1 = Function(VV)
u0,p0 = split(up0)
u1,p1 = split(up1)
du,dp = TrialFunctions(VV)
v,q= TestFunctions(VV)

time = 0.05
k = 0.002
dt = k
B=Constant(("0.0","0.0","-14224.5"))
T=Constant(("0.0","0.0","-5000"))

#konst.
l = Constant(0.0105*1e9)
m = Constant(0.00230*1e9)
rho = Constant(1450)

#kinematics
d = u0.geometric_dimension()
I=Identity(d)
F1 = I +0.5*(grad(u0)+grad(u1))
C=F1.T*F1
E = variable(0.5*(C-I))
W = 0.5*l*(tr(E))**2 + m*tr(E*E)
S=diff(W,E)
P=F1*S

F = rho*inner((p1-p0),v)*dx +dt*inner(P,grad(v))*dx +inner(u1-u0,q)*dx -dt *inner(0.5*(p1+p0),q)*dx-dt*inner(B,v)*dx - dt*inner(T,v)*ds(1) 
t = dt

u0=Constant((0.0,0.0,0.0))
p0=Constant((0.0,0.0,0.0))

bcu = DirichletBC(V, Constant(("0.0","0.0","0.0")), "x[0]<0.1*0.001 + DOLFIN_EPS" )
bcp= DirichletBC(V, Constant(("0.0","0.0","0.0")), "x[0]<0.1*0.001 + DOLFIN_EPS" )
bc = [bcu,bcp]

while t<=time:
    solve(F==0,up1,bc)

    t=t+dt
    up0.assign(up1)


    

