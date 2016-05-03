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


"""
terminal>> python Exercise2.py 
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
Solving linear variational problem.
**********************************
Polynomial order = 1

-------------------lam=1-------------------

L2 error:  [0.07143564353036871, 0.01858894010975571, 0.004697079781323455, 0.001177468248156252]

H1 error:  [1.3517242507219713, 0.6790867417407639, 0.3398473366000941, 0.1699577953467868]

L2 con-rate=1.975326 C=2.213178

H1 con-rate=0.997337 C=7.621186

---------------------------------------------
-------------------lam=100-------------------

L2 error:  [0.29985169058331784, 0.16406710708851321, 0.06076135418164534, 0.017653643817245494]

H1 error:  [2.630391833510895, 1.423505893625182, 0.5774530212031616, 0.2179137174299302]

L2 con-rate=1.369169 C=3.795261

H1 con-rate=1.208202 C=23.611052

---------------------------------------------
-------------------lam=10000-------------------

L2 error:  [0.444750039708708, 0.45628947339193204, 0.43298295714847634, 0.3519439457224945]

H1 error:  [3.7443848306183782, 3.6589796437836535, 3.4092084266679863, 2.7338766907550016]

L2 con-rate=0.108859 C=0.567093

H1 con-rate=0.146335 C=5.043682

---------------------------------------------


**********************************
Polynomial order = 2

-------------------lam=1-------------------

L2 error:  [0.0020804727431950787, 0.0002521449774644887, 3.124455010249258e-05, 3.896910893232253e-06]

H1 error:  [0.12333779284666148, 0.031097865213979455, 0.007791325005348004, 0.0019488967225476028]

L2 con-rate=3.019367 C=0.386373

H1 con-rate=1.994832 C=3.920336

---------------------------------------------
-------------------lam=100-------------------

L2 error:  [0.014397270633258623, 0.0014981362081414133, 0.00011945190272137094, 8.7436893209144e-06]

H1 error:  [0.45195247197065147, 0.09708626269773965, 0.01665272098869556, 0.0028154805090628452]

L2 con-rate=3.570446 C=7.716594

H1 con-rate=2.452345 C=33.981253

---------------------------------------------
-------------------lam=10000-------------------

L2 error:  [0.029893235368636906, 0.007174945881791682, 0.0015772288060908345, 0.00027219372347456357]

H1 error:  [0.9452508877864363, 0.4515093401616464, 0.19729045871526465, 0.06764907638113875]

L2 con-rate=2.252270 C=1.596042

H1 con-rate=1.260810 C=9.058619

---------------------------------------------


"""


