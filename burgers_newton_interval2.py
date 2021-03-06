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
    ic2 = project(Expression("sin(2*pi*x[0])"),  V, annotate=True, name="ic2")
    #ic2 = project(Expression("0.0"),  V, annotate=True, name="ic2")

    # Run the first interval, starting from ic1
    adj_start_timestep(0.0) 
    u = main(ic2, 0.2, 0.4, annotate=True)

    J = Functional(u**2*dx*dt)

    ctrls = Control(ic2)
    
    rf = ReducedFunctional(J, ctrls)
    print "Evaluate functional at ic2", rf([ic2])
    grad = rf.derivative(forget=False)
    print norm(grad[0])
