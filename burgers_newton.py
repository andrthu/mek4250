"""
Implementation of Burger's equation with nonlinear solve in each
timestep
"""

import sys

from dolfin import *
from dolfin_adjoint import *

dolfin.parameters["adjoint"]["cache_factorizations"] = True

if dolfin.__version__ > '1.2.0':
    dolfin.parameters["adjoint"]["symmetric_bcs"] = True

n = 4
mesh = UnitIntervalMesh(n)
V = FunctionSpace(mesh, "CG", 2)

def Dt(u, u_, timestep):
    return (u - u_)/timestep

def main(ic, start, end, annotate=False):

    u_ = ic.copy(deepcopy=True, name="Velocity".format(start), annotate=annotate)
    u = Function(V, name="VelocityNext".format(start))
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(end-start)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = start
    while (t < end - DOLFIN_EPS):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

        adj_inc_timestep(t)
        print "Solved for time ", t

    return u_

if __name__ == "__main__":
    # Dont print out FEniCS solver messages
    set_log_level(ERROR)
 
    # Dummy variables - its not being used
    #project(Constant(0), V, name="ic1", annotate=True)
    #project(Constant(0), V, name="ic2", annotate=True)
    #project(Constant(0), V, name="PreviousVelocity", annotate=True)


 
    # TODO: Should read this from file
    ic1 = project(Expression("sin(2*pi*x[0])"),  V, annotate=True, name="ic1")
    
    ic2 = project(Expression("100*sin(2*pi*x[0])"),  V, annotate=True, name="ic2")
    ic3 = project(Constant(0.0),  V, annotate=True, name="ic3")

    # Run the first interval, starting from ic1
    adj_start_timestep(0.0) 
    u1 = main(ic1, 0.0, 0.2, annotate=True)

    J1 = Functional(u1*dx*dt + (u1-ic2)**2*dx*dt[FINISH_TIME])
    

    

    ctrls = Control(ic1)
    
    rf = ReducedFunctional(J1, ctrls)
    print "Evaluate functional at ic1", rf([ic1])
    grad = rf.derivative(forget=False)
    print norm(grad[0])


    #minimize(rf, method="L-BFGS-B")
    
    u2 = main(ic3, 0.2, 0.4, annotate=True)

    J2 = Functional(u2**2*dx*dt)

    ctrls2 = Control(ic3)

    rf2 = ReducedFunctional(J2, ctrls2)
    print "Evaluate functional at ic2", rf2([ic3])
    grad = rf2.derivative(forget=False)
    print norm(grad[0])



    """
    # Run the first interval, starting from ic1
    adj_start_timestep(0.0) 
    ic1 = project(Expression("sin(2*pi*x[0])"),  V, annotate=False, name="ic1")
    u = main(ic1, 0.0, 0.2, annotate=True)

    #forward1_final = project(Constant(10), V, name="forward1_final", annotate=True)
    #print "forward 1 final is at time "
 
    print "Starting new interval"

    old_u = u.copy(deepcopy=True, name="PreviousVelocity")


    # Run the second interval, starting from ic2
    ic2 = project(Expression("sin(2*pi*x[0])"),  V, annotate=False, name="ic2")
    u = main(ic2, 0.2, 0.4, annotate=True)

    adj_html("forward.html", "forward")
    J = Functional((old_u-ic2)**2*dx*dt[FINISH_TIME])

    ctrls = [Control(ic2)]
    
    rf = ReducedFunctional(J, ctrls)
    print "Evaluate functional at ic2", rf([ic2])
    grad = rf.derivative(forget=False)
    print norm(grad[0])
    #minimize(rf, method="L-BFGS-B")
    """
