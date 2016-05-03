from dolfin import *
from numpy import pi,matrix,sqrt,diagflat,zeros,vstack,ones,log,array,size,exp
from scipy import linalg
import matplotlib.pyplot as plt

#defining lists with h values and lambdas, and setting my=1
Lam = [1,100,10000]
my = 1
H = [8,16,32,64]
#Expression u exact
ue = Expression(("pi*x[0]*cos(pi*x[0]*x[1])","-pi*x[1]*cos(pi*x[0]*x[1])"),degree=3)

#source term found by my*laplace(ue)
f = Expression(("(-pow(pi*x[0],3)*cos(pi*x[0]*x[1]) - pow(pi,3)*x[0]*pow(x[1],2)*cos(pi*x[0]*x[1]) - 2*pow(pi,2)*x[1]*sin(pi*x[0]*x[1]))",  "pow(pi,3)*pow(x[0],2)*x[1]*cos(pi*x[0]*x[1]) + 2*pow(pi,2)*x[0]*sin(pi*x[0]*x[1]) + pow(pi*x[1],3)*cos(pi*x[0]*x[1])"),degree=2)

#lists to store errors
L2_error = [[[],[],[]],[[],[],[]]]
H1_error = [[[],[],[]],[[],[],[]]]

#lists to store convergence rates aqnd their constants
con = [[[],[]],[[],[]]]

#solving the equation starts here. Loop over two types of element degrees
for p in [1,2]:
    #loop over our lambda values
    for i in range(len(Lam)):
        #define a list to store minimum mesh resolution
        hv = []
        
        #loop over different mesh resolutions
        for j in range(len(H)):
            
            #defone our mesh
            mesh = UnitSquareMesh(H[j],H[j])
            
            #define our vectorspace, and an extra space to measure error
            V = VectorFunctionSpace(mesh,"Lagrange",p)
            V2= VectorFunctionSpace(mesh,"Lagrange",p+3)

            
            #test and trial
            u = TrialFunction(V)
            v = TestFunction(V)
            #fenics function that is our current lambda
            l = Constant(Lam[i])
            
            #interpolta our source term into our space
            F = -interpolate(f,V)
            
            #write up our variational form
            a = inner(grad(u),grad(v))*dx + l*div(u)*div(v)*dx
            L = dot(F,v)*dx
            
            
            #fix our Dirichlet BCs
            bc = DirichletBC(V,ue,"on_boundary")

            #solve our equation
            u = Function(V)
            solve(a==L,u,bc)

            #interpolate the exact solution to a our higher degree space
            Ue = interpolate(ue,V2)
            
            #find the errors
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


        #plt.plot(log(array(hv)),log(array(L2_error[p-1][i])))
        #plt.plot(log(array(hv)),log(array(H1_error[p-1][i])))
        #plt.plot((log(array(hv))),p*(log(array(hv))),"r--")
        #plt.plot((log(array(hv))),(p+1)*(log(array(hv))),"y--")
        #plt.legend(["L2","H1",str(p)+"*log(h)",str(2*p)+"*log(h)" ])
        #plt.show()


#do some fancy output print
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

#plot to justify least squares.
for j in range(3):
    fig,ax = plt.subplots(2, 1,sharex=True)
    
    for l in range(2):
        p=l+1
        ax[l].plot(log(array(hv)),log(array(L2_error[l][j])))
        ax[l].plot(log(array(hv)),log(array(H1_error[l][j])))
        ax[l].plot((log(array(hv))),(l+1)*(log(array(hv))),"g--")
        ax[l].plot((log(array(hv))),(l+2)*(log(array(hv))),"b--")
        ax[l].set_title("P"+str(p)+" element, lambda="+str(Lam[j]))
        ax[l].legend(["L2","H1",str(p)+"*log(h)",str(l+2)+"*log(h)" ],loc=4)
        ax[l].set_xlabel("log(h)")
        ax[l].set_ylabel("log(error)")
    plt.show()


