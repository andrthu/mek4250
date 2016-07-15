import sys

from dolfin import *
from dolfin_adjoint import *

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
 
    # TODO: Should read this from file
    ic1 = project(Expression("sin(2*pi*x[0])"),  V, annotate=True, name="ic1")
    ic2 = project(Expression("sin(2*pi*x[0])"),  V, annotate=True, name="ic2")

    # Run the first interval, starting from ic1
    adj_start_timestep(0.0) 
    u = main(ic1, 0.0, 0.2, annotate=True)

    J = Functional(u*dx*dt + (u-ic2)**2*dx*dt[FINISH_TIME])

    ctrls = Control(ic1)
    
    rf = ReducedFunctional(J, ctrls)
    print "Evaluate functional at ic1", rf([ic1])
    grad = rf.derivative(forget=False)
    print norm(grad[0])
    #minimize(rf, method="L-BFGS-B")
