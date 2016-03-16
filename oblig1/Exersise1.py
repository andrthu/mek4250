from dolfin import *
from numpy import pi,matrix, sqrt, diagflat,zeros,vstack,ones,log,array,size
#from scipy.special import factorial as fac
from math import factorial as fac
from scipy import linalg


def Hp_norm(p,k,l):
    s = 0

    for i in range(p+1):
        for j in range(i+1):
            s = s + (k*pi)**(2*(i-j))*(l*pi)**(2*j)*fac(i)/(fac(j)*fac(i-j)) 
        
    return sqrt(0.25*s)


def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False

L2 = [[],[]]
H1 = [[],[]]
h_val = [[],[]]
con = [[],[]]

for p in [1,2]:
    for k in [1,10,100]:
        for l in [1,10,100]:

            l2 = []
            h1 = []
            hv = []
            for h in [8,16,32,64]:

                
                mesh = UnitSquareMesh(h,h)

                V = FunctionSpace(mesh,'Lagrange',p)
                V2 = FunctionSpace(mesh,'Lagrange',p+3)
                

                f = Expression('pi*pi*sin(%e*pi*x[0])*cos(%e*pi*x[1])*%e'
                               %(k,l,l**2+k**2))

                ue = Expression('sin(%e*pi*x[0])*cos(%e*pi*x[1])'%(k,l))
                g = Constant(0)
                
                

                u = TrialFunction(V)
                v = TestFunction(V)

                a = inner(grad(u),grad(v))*dx
                L = f*v*dx 

                
                
                bc = DirichletBC(V,g,Dirichlet_boundary)
                
                U = Function(V)

                solve(a==L,U,bc,solver_parameters={"linear_solver": "cg"})
                Ue = interpolate(ue,V2)

                A =errornorm(U,Ue)
                B = errornorm(U,Ue,'H1')

                L2[p-1].append(A/Hp_norm(p+1,k,l))
                H1[p-1].append(B/Hp_norm(p+1,k,l))
                h_val[p-1].append( mesh.hmin())
                
                l2.append(A)
                h1.append(B)
                hv.append(mesh.hmin())
                #plot(U)
                #interactive()
            Q1 = vstack([log(array(hv)),ones(len(hv))]).T
            con[0].append(linalg.lstsq(Q1, log(array(l2)))[0])
            con[1].append(linalg.lstsq(Q1, log(array(h1)))[0])

Q1 = vstack([log(array(h_val[0])),ones(len(h_val[0]))]).T
print linalg.lstsq(Q1, log(array(L2[0])))[0]
print linalg.lstsq(Q1, log(array(H1[0])))[0]

Q2 = vstack([log(array(h_val[1])),ones(len(h_val[1]))]).T
print linalg.lstsq(Q2, log(array(L2[1])))[0]
print linalg.lstsq(Q2, log(array(H1[1])))[0]

print

print L2[0]
print
print L2[1]
print 
print con[0]
print
print con[1]




