from dolfin import *
from numpy import pi,matrix,sqrt,diagflat,zeros,vstack,ones,log,array,size,exp
from scipy import linalg
import matplotlib.pyplot as plt

Lam = [1,100,10000]
my = 1
H = [8,16,32,64]
ue = Expression(("pi*x[0]*cos(pi*x[0]*x[1])","-pi*x[1]*cos(pi*x[0]*x[1])"),degree=3)

f = Expression(("(-pow(pi*x[0],3)*cos(pi*x[0]*x[1]) - pow(pi,3)*x[0]*pow(x[1],2)*cos(pi*x[0]*x[1]) - 2*pow(pi,2)*x[1]*sin(pi*x[0]*x[1]))",  "pow(pi,3)*pow(x[0],2)*x[1]*cos(pi*x[0]*x[1]) + 2*pow(pi,2)*x[0]*sin(pi*x[0]*x[1]) + pow(pi*x[1],3)*cos(pi*x[0]*x[1])"),degree=2)

L2_error = [[[],[],[]],[[],[],[]]]
H1_error = [[[],[],[]],[[],[],[]]]

con = [[[],[]],[[],[]]]

for i in range(len(Lam)):
    hv = []
    for j in range(len(H)):
        
        mesh = UnitSquareMesh(H[j],H[j])
        
        V = VectorFunctionSpace(mesh,"Lagrange",2)
        V2 = VectorFunctionSpace(mesh,"Lagrange",2+3)

        Q  = FunctionSpace(mesh,"Lagrange",1)
        W = V*Q
        
        F = -interpolate(f,V)

        u,p=TrialFunctions(W)
        v,q=TestFunctions(W)
        
        l = Constant(1./Lam[i])
        
        a = (inner(grad(u),grad(v)) + p*div(v) + q*div(u)-l*p*q)*dx
        L = inner(F,v)*dx

        bc = DirichletBC(W.sub(0),ue,"on_boundary")

        Phi = Function(W)

        solve(a==L,Phi,bc)

        U,P = Phi.split()

        Ue = interpolate(ue,V2)

        L2_error[0][i].append(errornorm(Ue,U))
        H1_error[0][i].append(errornorm(Ue,U,'H1'))
        hv.append(mesh.hmax())
        
    Q1 = vstack([log(array(hv)),ones(len(hv))]).T
        
    con[0][0].append(linalg.lstsq(Q1, log(array(L2_error[0][i])))[0])
    con[0][1].append(linalg.lstsq(Q1, log(array(H1_error[0][i])))[0])

for i in range(1):
    print "**********************************"
    p=i+1
    print "Polynomial order = %d" %p
    print
    for j in range(len(Lam)):
        print "-------------------lam=%d-------------------" % Lam[j]
        print
        print "L2 error: ", L2_error[i][j]
        print
        print "H1 error: ", H1_error[i][j]
        print
        print "L2 con-rate=%f C=%f" % (con[i][0][j][0] , exp(con[i][0][j][1]))
        print
        print "H1 con-rate=%f C=%f" % (con[i][1][j][0] , exp(con[i][1][j][1]))
        print
        print "-------------------------------------------"
    print
    print
