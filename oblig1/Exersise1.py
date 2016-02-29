from dolfin import *


def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False


for p in [2]:
    for k in [100]:
        for l in [100]:
            for h in [64]:


                mesh = UnitSquareMesh(h,h)

                V = FunctionSpace(mesh,'Lagrange',p)
                V2 = FunctionSpace(mesh,'Lagrange',p+3)


                f = Expression('pi*pi*sin(%e*pi*x[0])*cos(%e*pi*x[1])*%e'
                               %(k,l,l**2+k**2))

                g = Constant(0)
                #h = Expression('x[1]')

                u = TrialFunction(V)
                v = TestFunction(V)

                a = inner(grad(u),grad(v))*dx
                L = f*v*dx 

                print 1./mesh.hmin()
                
                bc = DirichletBC(V,g,Dirichlet_boundary)

                U = Function(V)

                solve(a==L,U,bc,solver_parameters={"linear_solver": "cg"})

                plot(U)
                interactive()
