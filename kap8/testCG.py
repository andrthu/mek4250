from dolfin import *
import time
import matplotlib.pyplot as plt
from numpy import array






def solving_time(A,b,solver,V):
    U = Function(V)

    PETScOptions().set("ksp_type", "cg")
    PETScOptions().set("pc_type", solver[1])

    Loser = PETScKrylovSolver()
    
    
    t0=time.time()
    Loser.solve(A,U.vector(), b)
    t1=time.time()
    
    return t1-t0

def amg_solving_time(A,b,solver,V):

    U = Function(V)
    t0=time.time()
    solve(A,U.vector(),b,solver[0],solver[1])
    t1=time.time()
    
    return t1-t0

cg_time     = [[],[]]
amg_time    = [[],[]]
ilu_time    = [[],[]]
jacobi_time = [[],[]]

Time = [cg_time,jacobi_time,amg_time,ilu_time]


def test_PC(pc,N):
    
    #N = [16,32,64,128,256,512]

    M = [UnitIntervalMesh,UnitSquareMesh]

    Time = [[],[]]
    
    for i in range(len(M)):

    
        for j in range(len(N)):

            if i==0:
                mesh = M[i](N[j])
            else:
                mesh = M[i](N[j],N[j])

            V = FunctionSpace(mesh,"Lagrange",1)

            bc=DirichletBC(V,Constant(0),"on_boundary")

            
            u=TrialFunction(V)
            v=TestFunction(V)

            a=inner(grad(u),grad(v))*dx
            if i==0:
                f=Expression("pow(pi,2)*sin(pi*x[0])")            
            else:
                f=Expression("2*pow(pi,2)*sin(pi*x[0])*sin(pi*x[1])")
            
            L=f*v*dx

            A,b = assemble_system(a,L,bc)

            if pc[1] == 'amg':
                t2 = amg_solving_time(A,b,pc,V)
                Time[i].append(t2)
            else:
            
                t2 = solving_time(A,b,pc,V)
                Time[i].append(t2)
    return Time

Solver=[["cg","none"],["cg","jacobi"],["cg","amg"],["cg","ilu"]]
N = [16,32,64,128,256,512]
amg_time    = test_PC(Solver[2],N)
cg_time     = test_PC(Solver[0],N)
ilu_time    = test_PC(Solver[3],N)
jacobi_time = test_PC(Solver[1],N)


list_krylov_solver_preconditioners()

plt.plot(array(N),array(cg_time[0]))
plt.plot(array(N),array(amg_time[0]))
plt.plot(array(N),array(ilu_time[0]))
plt.plot(array(N),array(jacobi_time[0]))

plt.legend(['cg','amg','ilu','jacobi'],loc=2)
plt.xlabel('dofs')
plt.ylabel('time in seconds')
plt.show()


plt.plot(array(N)**2,array(cg_time[1]))
plt.plot(array(N)**2,array(amg_time[1]))
plt.plot(array(N)**2,array(ilu_time[1]))
plt.plot(array(N)**2,array(jacobi_time[1]))

plt.legend(['cg','amg','ilu','jacobi'],loc=2)
plt.xlabel('dofs')
plt.ylabel('time in seconds')
plt.show()

"""
for i in range(len(M)):

    
    for j in range(len(N)):

        if i==0:
            mesh = M[i](N[j])
        else:
            mesh = M[i](N[j],N[j])

        V = FunctionSpace(mesh,"Lagrange",1)

        bc=DirichletBC(V,Constant(0),"on_boundary")

            
        u=TrialFunction(V)
        v=TestFunction(V)

        a=inner(grad(u),grad(v))*dx
        if i==0:
            f=Expression("pow(pi,2)*sin(pi*x[0])")            
        else:
            f=Expression("2*pow(pi,2)*sin(pi*x[0])*sin(pi*x[1])")
            
        L=f*v*dx

        A,b = assemble_system(a,L,bc)

        for k in range(len(Time)):
            
            t2 = solving_time(A,b,Solver[k])
            Time[k][i].append(t2)
        
        
        t2=solving_time(A,b,Solver[0])
        print "cg",t2
        cg_time[i].append(t2)
        try:
            t2=solving_time(A,b,Solver[1])
            print "jacobi",t2
            jacobi_time[i].append(t2)
        except:
            jacobi_time[i].append(0)
            
        t2=solving_time(A,b,Solver[2])
        print "amg",t2
        amg_time[i].append(t2)

        t2=solving_time(A,b,Solver[3])
        print "ilu",t2
        ilu_time[i].append(t2)
        
"""




