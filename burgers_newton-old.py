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

    u_ = ic.copy(deepcopy=True, name="Velocity{}".format(start), annotate=annotate)
    u = Function(V, name="VelocityNext{}".format(start))
    v = TestFunction(V)

    nu = Constant(0.0001)

    timestep = Constant(1.0/n)

    F = (Dt(u, u_, timestep)*v
         + u*u.dx(0)*v + nu*u.dx(0)*v.dx(0))*dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    t = start
    while (t <= end):
        solve(F == 0, u, bc, annotate=annotate)
        u_.assign(u, annotate=annotate)

        t += float(timestep)

        adj_inc_timestep(t)
        print t

    return u_

if __name__ == "__main__":
    # Dont print out FEniCS solver messages
    set_log_level(ERROR)
 
    # Dummy variables - its not being used
    project(Constant(0), V, name="forward1_final", annotate=True)
    project(Constant(0), V, name="ic2", annotate=True)


    # Run the first interval, starting from ic1
    adj_start_timestep(0.0) 
    ic1 = project(Expression("sin(2*pi*x[0])"),  V, annotate=False, name="ic1")
    forward1 = main(ic1, 0.0, 0.2, annotate=True)

    forward1_final = project(Constant(10), V, name="forward1_final", annotate=True)
    print "forward 1 final is at time "

    # Run the second interval, starting from ic2
    ic2 = project(Expression("sin(2*pi*x[0])"),  V, annotate=True, name="ic2")
    forward2 = main(ic2, 0.2, 0.4, annotate=True)

    adj_html("forward.html", "forward")
    J = Functional((forward1_final-ic2)**2*dx*dt[0.25])
    J = Functional((forward1_final)**2*dx*dt[0.3])

    ctrls = [Control(ic2)]
    
    rf = ReducedFunctional(J, ctrls)
    print "Evaluate functional at ic2", rf([ic2])
    rf.derivative()
    #minimize(rf, method="L-BFGS-B")
