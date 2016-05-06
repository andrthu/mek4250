from dolfin import *
import time
import matplotlib.pyplot as plt
from numpy import array

def solving_time(A,b,solver):
    U = Function(V)
    
    t0=time.time()
    
    if len(solver)==2:
        solve(A,U.vector(),b,solver[0],solver[1])
    else:
        solve(A,U.vector(),b,solver[0])
    t1=time.time()
    
    return t1-t0,U


Solver=[["cg"],["cg","amg"],["cg","ilu"],["gmres"],["gmres","ilu"],["gmres","amg"],["bicgstab"],["bicgstab","ilu"],["bicgstab","amg"]]

h = [8,15,32,64]

C= [1,10,100,1000,10000]

ue = Expression("sin(pi*x[0])*sin(pi*x[1])")



cg_time = [[],[],[],[],[]]
amg_time = [[],[],[],[],[]]
ilu_time = [[],[],[],[],[]]

for i in range(len(C)):
    
    f = Expression("pow(pi,2)*2*(sin(pi*x[0])*sin(pi*x[1])) + %d*pi*sin(7*pi*x[0])*cos(pi*x[1])*sin(pi*x[0])"%C[i])
    g = Expression("%d*sin(7*pi*x[0])"%C[i])

    

    for j in range(len(h)):
        

        mesh = UnitSquareMesh(h[j],h[j])

        V = FunctionSpace(mesh,"Lagrange",1)

        u=TrialFunction(V)
        v=TestFunction(V)
        
        
        a = inner(u,v)*dx + g*v*u.dx(1)*dx
        L = f*v*dx

        bc = DirichletBC(V,ue,"on_boundary")

        
        
        A,b=assemble_system(a,L,bc)

        t2,U=solving_time(A,b,Solver[0+2*3])
        print "cg",t2
        cg_time[i].append(t2)

        t2,U=solving_time(A,b,Solver[1+2*3])
        print "amg",t2
        amg_time[i].append(t2)
        
        t2,U=solving_time(A,b,Solver[2+2*3])
        print "ilu",t2
        ilu_time[i].append(t2)
plt.plot(array(N),array(cg_time[0]))
plt.plot(array(N),array(amg_time[0]))
plt.plot(array(N),array(ilu_time[0]))
plt.legend(['cg','amg','ilu'])
plt.xlabel('dofs')
plt.ylabel('time in seconds')
plt.show()
