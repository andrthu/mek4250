from dolfin import *
import time
import matplotlib.pyplot as plt
from numpy import array

#parameters["maximum_iterations"]=5000
def solving_time(A,b,solver):
    U = Function(V)
    
    t0=time.time()
    
    if len(solver)==2:
        solver = KrylovSolver(solver[0],solver[1])
        #solve(A,U.vector(),b,solver[0],solver[1])
    else:
        solver = KrylovSolver(solver[0])

    
        #solve(A,U.vector(),b,solver[0])
    #solver.parameters["maximum_iterations"]=1000
    
    solver.solve(A,U.vector(),b)
    t1=time.time()
    
    return t1-t0,U


Solver=[["cg"],["cg","amg"],["cg","ilu"],["gmres"],["gmres","ilu"],["gmres","amg"],["bicgstab"],["bicgstab","ilu"],["bicgstab","amg"]]

h = [8,15,32,64]

C= [10,100,1000,10000]

ue = Expression("sin(pi*x[0])*sin(pi*x[1])")



cg_time = [[],[],[],[],[]]
amg_time = [[],[],[],[],[]]
ilu_time = [[],[],[],[],[]]

gmres_time = [[],[],[],[],[]]
g_amg_time = [[],[],[],[],[]]
g_ilu_time = [[],[],[],[],[]]

bicgstab_time = [[],[],[],[],[]]
b_amg_time = [[],[],[],[],[]]
b_ilu_time = [[],[],[],[],[]]

for i in range(len(C)):
    
    f = Expression("pow(pi,2)*2*(sin(pi*x[0])*sin(pi*x[1])) + %d*pi*sin(7*pi*x[0])*cos(pi*x[1])*sin(pi*x[0])"%C[i])
    g = Expression("%d*sin(7*pi*x[0])"%C[i])

    f2=Constant(0)

    for j in range(len(h)):
        

        mesh = UnitSquareMesh(h[j],h[j])

        V = FunctionSpace(mesh,"Lagrange",1)

        u=TrialFunction(V)
        v=TestFunction(V)
        
        
        a = inner(grad(u),grad(v))*dx + g*v*u.dx(1)*dx
        L = f*v*dx

        bc = DirichletBC(V,ue,"on_boundary")

        
        
        A,b=assemble_system(a,L,bc)
        try:
            t2,U=solving_time(A,b,Solver[0])
            print "cg",t2
            cg_time[i].append(t2)
        except:
            cg_time[i].append(0)
        try:
            t2,U=solving_time(A,b,Solver[1])
            print "amg",t2
            amg_time[i].append(t2)
        except:   
            amg_time[i].append(0)

        try:
            t2,U=solving_time(A,b,Solver[2])
            print "ilu",t2
            ilu_time[i].append(t2)
        except:
            ilu_time[i].append(0)

        try:
            t2,U=solving_time(A,b,Solver[0+3])
            print "gmres",t2
            gmres_time[i].append(t2)
        except:
            gmres_time[i].append(0)
        try:
            t2,U=solving_time(A,b,Solver[1+3])
            print "g_amg",t2
            g_amg_time[i].append(t2)
        except:   
            g_amg_time[i].append(0)

        try:
            t2,U=solving_time(A,b,Solver[2+3])
            print "g_ilu",t2
            g_ilu_time[i].append(t2)
        except:
            g_ilu_time[i].append(0)

        try:
            t2,U=solving_time(A,b,Solver[0+6])
            print "bicgstab",t2
            bicgstab_time[i].append(t2)
        except:
            bicgstab_time[i].append(0)
        try:
            t2,U=solving_time(A,b,Solver[1+6])
            print "b_amg",t2
            b_amg_time[i].append(t2)
        except:   
            b_amg_time[i].append(0)

        try:
            t2,U=solving_time(A,b,Solver[2+6])
            print "b_ilu",t2
            b_ilu_time[i].append(t2)
        except:
            b_ilu_time[i].append(0)
            
fig,ax = plt.subplots(2, 2)
for i in range(4):
    x = i/2
    y= i%2
    ax[x,y].plot(array(h)**2,array(cg_time[i]))
    ax[x,y].plot(array(h)**2,array(amg_time[i]))
    ax[x,y].plot(array(h)**2,array(ilu_time[i]))
    ax[x,y].legend(['cg','amg','ilu'],loc=2)
    ax[x,y].set_xlabel('dofs')
    ax[x,y].set_ylabel('time in seconds')
plt.show()

fig,ax = plt.subplots(2, 2)
for i in range(4):
    x = i/2
    y= i%2
    ax[x,y].plot(array(h)**2,array(gmres_time[i]))
    ax[x,y].plot(array(h)**2,array(g_amg_time[i]))
    ax[x,y].plot(array(h)**2,array(g_ilu_time[i]))
    ax[x,y].legend(['gmres','amg','ilu'],loc=2)
    ax[x,y].set_xlabel('dofs')
    ax[x,y].set_ylabel('time in seconds')
plt.show()

fig,ax = plt.subplots(2, 2)
for i in range(4):
    x = i/2
    y= i%2
    ax[x,y].plot(array(h)**2,array(bicgstab_time[i]))
    ax[x,y].plot(array(h)**2,array(b_amg_time[i]))
    ax[x,y].plot(array(h)**2,array(b_ilu_time[i]))
    ax[x,y].legend(['bicgstab','amg','ilu'],loc=2)
    ax[x,y].set_xlabel('dofs')
    ax[x,y].set_ylabel('time in seconds')
plt.show()
