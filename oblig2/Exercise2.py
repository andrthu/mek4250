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

for p in [1,2]:
    for i in range(len(Lam)):
        hv = []
        
        for j in range(len(H)):

            mesh = UnitSquareMesh(H[j],H[j])

            V = VectorFunctionSpace(mesh,"Lagrange",p)
            V2= VectorFunctionSpace(mesh,"Lagrange",p+3)


            u = TrialFunction(V)
            v = TestFunction(V)
            l = Constant(Lam[i])

            F = -interpolate(f,V)
            
            a = inner(grad(u),grad(v))*dx + l*div(u)*div(v)*dx
            L = dot(F,v)*dx

            u = Function(V)
            bc = DirichletBC(V,ue,"on_boundary")
            
            solve(a==L,u,bc)
            Ue = interpolate(ue,V2)

            L2_error[p-1][i].append(errornorm(Ue,u))
            H1_error[p-1][i].append(errornorm(Ue,u,'H1'))
            hv.append(mesh.hmax())
            #plot(u-Ue)
            #interactive()
            #plot(Ue)
            #interactive()

        #calculate convergence using least square for each set of
        #parameters.
        Q1 = vstack([log(array(hv)),ones(len(hv))]).T
        
        con[p-1][0].append(linalg.lstsq(Q1, log(array(L2_error[p-1][i])))[0])
        con[p-1][1].append(linalg.lstsq(Q1, log(array(H1_error[p-1][i])))[0])


        plt.plot(log(array(hv)),log(array(L2_error[p-1][i])))
        plt.plot(log(array(hv)),log(array(H1_error[p-1][i])))
        plt.plot((log(array(hv))),p*(log(array(hv))),"r--")
        plt.plot((log(array(hv))),(p+1)*(log(array(hv))),"y--")
        plt.legend(["L2","H1",str(p)+"*log(h)",str(2*p)+"*log(h)" ])
        #plt.show()
            
            
for i in range(2):
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
        print "---------------------------------------------"
    print
    print
        
"""
print L2_error
print
print H1_error
print
print con
"""
