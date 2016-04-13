from dolfin import *
import time
def heat2d(alfa,beta,N,T,N_t):

    mesh = UnitSquareMesh(N,N)
    DT = float(T)/N_t
    dt=Constant(DT)
    V = FunctionSpace(mesh,"Lagrange",2)
    V2 = FunctionSpace(mesh,'Lagrange',4)

    g = Expression("1+pow(x[0],2)+alfa*pow(x[1],2)+beta*t",alfa=alfa,beta=beta,t=0,degree=3)
    f = Expression("beta-2-2*alfa",alfa=alfa,beta=beta,degree=3)
    
    u0=project(g,V,solver_type="lu")
    u1=Function(V)
    print "u0",errornorm(g,u0)
    u=TrialFunction(V)
    v=TestFunction(V)

    a = u*v*dx+ dt*inner(grad(u),grad(v))*dx
    L = u0*v*dx+dt*f*v*dx

    bc=DirichletBC(V,g,"on_boundary")

    A = assemble(a)

    t = DT

    while t<=T:
        #plot(u0)
        #time.sleep(0.3)
        b = assemble(L)
        g.t=t
        bc.apply(A,b)
        solve(A,u1.vector(),b)
        u0.assign(u1)
        t +=DT
        
    #plot(u0)
    #interactive()
    
    G= interpolate(g,V2)
    print errornorm(G,u1)
    
def heat3d(alfa,beta,gamma,N,T,N_t):

    mesh = UnitCubeMesh(N,N,N)
    DT = float(T)/N_t
    dt=Constant(DT)
    V = FunctionSpace(mesh,"Lagrange",1)
    V2 = FunctionSpace(mesh,'Lagrange',4)

    g = Expression("1+pow(x[0],2)+alfa*pow(x[1],2)+gamma*pow(x[2],2)+beta*t",alfa=alfa,beta=beta,gamma=gamma,t=0,degree=3)
    f = Expression("beta-2-2*alfa-2*gamma",alfa=alfa,beta=beta,gamma=gamma,degree=3)

    u0=project(g,V)
    u1=Function(V)

    u=TrialFunction(V)
    v=TestFunction(V)

    a = u*v*dx+ dt*inner(grad(u),grad(v))*dx
    L = u0*v*dx+dt*f*v*dx

    bc=DirichletBC(V,g,"on_boundary")

    A = assemble(a)

    t = DT

    while t<=T:
        plot(u0)
        b = assemble(L)
        g.t=t
        bc.apply(A,b)
        solve(A,u1.vector(),b)

        t +=DT
        u0.assign(u1)
    plot(u0)
    g.t=T
    G= interpolate(g,V2)
    interactive()
    print errornorm(G,u1)


a=1
b=2
c=3
N=10
T = 1.8
N_t=20
#heat3d(a,b,c,N,T,N_t)
heat2d(a,b,N,T,N_t)
