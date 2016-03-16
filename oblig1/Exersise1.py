from dolfin import *
from numpy import pi,matrix,sqrt,diagflat,zeros,vstack,ones,log,array,size,exp
#from scipy.special import factorial as fac
from math import factorial as fac
from scipy import linalg


#This function is an implementation of the expression for the norm found
#in exerccise 1a.
def Hp_norm(p,k,l):
    s = 0

    for i in range(p+1):
        for j in range(i+1):
            s = s + (k*pi)**(2*(i-j))*(l*pi)**(2*j)*fac(i)/(fac(j)*fac(i-j)) 
        
    return sqrt(0.25*s)


#handeling the boundary, which is dirichlet only for
#x=0 and x=1
def Dirichlet_boundary(x, on_boundary):
    if on_boundary:
        if x[0] == 0 or x[0] == 1:
            return True
        else:
            return False
    else:
        return False

#lots of lists to store errors and convergence reates;
L2 = [[],[]]
H1 = [[],[]]
h_val = [[],[]]
con = [[ [],[] ],[ [],[] ]]

L2_print =[[[],[],[]],[[],[],[]]]
H1_print =[[[],[],[]],[[],[],[]]]


#solving the equation for a lot of different parameters, p being order of
#elements and k and l having the same meaning as in exercise
for p in [1,2]:
    for k in [1,10,100]:
        for l in [1,10,100]:
            
            #even more lists to store errors
            l2 = []
            h1 = []
            hv = []

            #Size of mesh parameter
            for h in [8,16,32,64]:

                #define mesh
                mesh = UnitSquareMesh(h,h)
                

                #define the functionspace and the space where the
                #exact solution should be
                V = FunctionSpace(mesh,'Lagrange',p)
                V2 = FunctionSpace(mesh,'Lagrange',p+3)
                
                #f= -laplace(ue)
                f = Expression('pi*pi*sin(%e*pi*x[0])*cos(%e*pi*x[1])*%e'
                               %(k,l,l**2+k**2))
                
                #ue given
                ue = Expression('sin(%e*pi*x[0])*cos(%e*pi*x[1])'%(k,l))
                #Dirichlet boundary function
                g = Constant(0)
                
                
                # define the weak formulation
                u = TrialFunction(V)
                v = TestFunction(V)

                a = inner(grad(u),grad(v))*dx
                L = f*v*dx 

                
                #give the boundary
                bc = DirichletBC(V,g,Dirichlet_boundary)
                
                #solve the equation
                U = Function(V)
                solve(a==L,U,bc,solver_parameters={"linear_solver": "cg"})

                #interpolate the exact solutionb into the V2 space
                Ue = interpolate(ue,V2)

                #measure the error in L2 and h1 norm
                A =errornorm(U,Ue)
                B = errornorm(U,Ue,'H1')

                #store the error in different ways
                L2[p-1].append(A/Hp_norm(p+1,k,l))
                H1[p-1].append(B/Hp_norm(p+1,k,l))
                h_val[p-1].append( mesh.hmin())
                
                l2.append(A)
                h1.append(B)
                hv.append(mesh.hmin())
                if k==1:
                    L2_print[p-1][int(log(l)/log(10))].append(A)
                    H1_print[p-1][int(log(l)/log(10))].append(B)

                if k==1 and l==1 and h==64 and p==1:
                    plot(U)
                    interactive()
                if k==100 and l==100 and h==64 and p==1:
                    plot(U)
                    interactive()
            
            #calculate convergence using least square for each set of
            #parameters.
            Q1 = vstack([log(array(hv)),ones(len(hv))]).T
            con[p-1][0].append(linalg.lstsq(Q1, log(array(l2)))[0])
            con[p-1][1].append(linalg.lstsq(Q1, log(array(h1)))[0])


#calculate and print convergence for both p1 and p2 elements using all
#the results.
print "*****************************"
for i in range(2):

    Q1 = vstack([log(array(h_val[i])),ones(len(h_val[i]))]).T
    
    L2LS = linalg.lstsq(Q1, log(array(L2[i])))[0]
    H1LS = linalg.lstsq(Q1, log(array(H1[i])))[0]
    print 
    print "----------------oreder =%d----------------------- "%( i+1)
    print "L2 convergence =%f and constant=%f" %(L2LS[0],exp(L2LS[1]))
    print
    print "H1 convergence =%f and constant=%f" %(H1LS[0],exp(H1LS[1]))
print "*****************************"
print

#print out the errrors for k=1
for i in range(2):
    print
    print "Error for element order %d" %(i+1)
    print
    for j in range(3):
        print"--------------------"
        print "<<<<<<<error for k=1 and l=%d>>>>>>>" % 10**j
        print
        print "L2 error: ", L2_print[i][j]
        print
        print "H1 error: ", H1_print[i][j]
        print"--------------------"
print

for i in range(2):
    print "****************************"
    print "convergence rates for elements of order %d" %(i+1) 
    print "****************************"
    for j in range(9):
        cL2 =con[i][0][j]
        cH1 =con[i][1][j]
        print "------------------"
        print "k=%d and l=%d" %(10**(j%3),10**(j/3))
        print 
        print "L2 convergence =%f and constant=%f" %(cL2[0],exp(cL2[1]))
        print 
        print "H1 convergence =%f and constant=%f" %(cH1[0],exp(cH1[1]))
        print "------------------"
        
        




"""
terminal> python Exercise1.py
*****************************

----------------oreder =1----------------------- 
L2 convergence =1.776496 and constant=0.041508

H1 convergence =0.911748 and constant=0.181145

----------------oreder =2----------------------- 
L2 convergence =2.570886 and constant=0.000820

H1 convergence =1.491997 and constant=0.003164
*****************************


Error for element order 1

--------------------
<<<<<<<error for k=1 and l=1>>>>>>>

L2 error:  [0.032766238358843174, 0.008462150927576493, 0.0021331628501450707, 0.0005344076037129104]

H1 error:  [0.43611616866425396, 0.218104587006901, 0.10904724617721379, 0.054522703124249276]
--------------------
--------------------
<<<<<<<error for k=1 and l=10>>>>>>>

L2 error:  [0.6722330538591342, 0.2446180327889003, 0.07860246093967867, 0.020908053540810826]

H1 error:  [15.899204235889268, 9.124783303768643, 4.615527564530228, 2.282328755498128]
--------------------
--------------------
<<<<<<<error for k=1 and l=100>>>>>>>

L2 error:  [191.99035176204282, 262.5175888557975, 3.0125832896210762, 4.6634965083137745]

H1 error:  [2761.707144295842, 3506.665166616119, 312.2994681818063, 432.522380201677]
--------------------

Error for element order 2

--------------------
<<<<<<<error for k=1 and l=1>>>>>>>

L2 error:  [0.0005687944394087238, 6.932977495255205e-05, 8.611112054164808e-06, 1.0752224227607405e-06]

H1 error:  [0.03314085488599447, 0.008386636058147628, 0.002105368509165348, 0.0005271586956191806]
--------------------
--------------------
<<<<<<<error for k=1 and l=10>>>>>>>

L2 error:  [0.3263570278412481, 0.025207667472855206, 0.002886293905081155, 0.00035056574226823776]

H1 error:  [8.791728328179158, 2.1875920019489614, 0.5710006773964853, 0.14461188934038433]
--------------------
--------------------
<<<<<<<error for k=1 and l=100>>>>>>>

L2 error:  [289.3910615468301, 91.85959608956985, 5.776418252203544, 1.7964749588148439]

H1 error:  [3775.694834292425, 1225.1062671080413, 553.7172353393878, 183.76591086295977]
--------------------

****************************
convergence rates for elements of order 1
****************************
------------------
k=1 and l=1

L2 convergence =1.980241 and constant=1.021877

H1 convergence =0.999942 and constant=2.466974
------------------
------------------
k=10 and l=1

L2 convergence =1.665838 and constant=12.995948

H1 convergence =0.938442 and constant=84.342502
------------------
------------------
k=100 and l=1

L2 convergence =2.253570 and constant=14999.996858

H1 convergence =1.151324 and constant=26031.249021
------------------
------------------
k=1 and l=10

L2 convergence =1.663255 and constant=12.830047

H1 convergence =0.944926 and constant=85.907784
------------------
------------------
k=10 and l=10

L2 convergence =1.190830 and constant=6.013075

H1 convergence =0.740449 and constant=98.137533
------------------
------------------
k=100 and l=10

L2 convergence =1.298707 and constant=358.875264

H1 convergence =0.576514 and constant=2884.015411
------------------
------------------
k=1 and l=100

L2 convergence =2.265711 and constant=15356.475296

H1 convergence =1.152285 and constant=26134.262986
------------------
------------------
k=10 and l=100

L2 convergence =1.367577 and constant=442.285878

H1 convergence =0.585193 and constant=2964.549325
------------------
------------------
k=100 and l=100

L2 convergence =2.302735 and constant=14499.701518

H1 convergence =1.116652 and constant=30563.304129
------------------
****************************
convergence rates for elements of order 2
****************************
------------------
k=1 and l=1

L2 convergence =3.015059 and constant=0.104979

H1 convergence =1.991671 and constant=1.048373
------------------
------------------
k=10 and l=1

L2 convergence =3.271422 and constant=83.036484

H1 convergence =1.971545 and constant=265.580690
------------------
------------------
k=100 and l=1

L2 convergence =2.598631 and constant=30848.599689

H1 convergence =1.422809 and constant=42797.018952
------------------
------------------
k=1 and l=10

L2 convergence =3.275512 and constant=83.936394

H1 convergence =1.967469 and constant=260.533783
------------------
------------------
k=10 and l=10

L2 convergence =2.874872 and constant=74.397249

H1 convergence =1.705687 and constant=375.021456
------------------
------------------
k=100 and l=10

L2 convergence =1.392888 and constant=327.276881

H1 convergence =0.766741 and constant=3781.919625
------------------
------------------
k=1 and l=100

L2 convergence =2.598767 and constant=30869.372506

H1 convergence =1.423052 and constant=42827.465163
------------------
------------------
k=10 and l=100

L2 convergence =1.388960 and constant=325.309665

H1 convergence =0.765313 and constant=3760.090085
------------------
------------------
k=100 and l=100

L2 convergence =2.721865 and constant=38942.470253

H1 convergence =1.413687 and constant=57275.977178
------------------


"""
