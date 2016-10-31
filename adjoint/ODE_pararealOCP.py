from my_bfgs.lbfgs import Lbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
from my_bfgs.steepest_decent import SteepestDecent,PPCSteepestDecent
from optimalContolProblem import OptimalControlProblem


class PararealOCP(OptimalControlProblem):

    def PC_maker(self,N,m):


        def pc(x):
            
            return x

        return pc

    def PPCSDsolve(self,N,m,my_list,x0=None,options=None):

        dt=float(self.T)/N
        if x0==None:
            x0 = np.zeros(N+m)
        
        result = []
        PPC = self.PC_maker(N,m)
        for i in range(len(my_list)):
        
            J,grad_J = self.generate_reduced_penalty(dt,N,m,my_list[i])

            self.update_SD_options(Lbfgs_options)
            SDopt = self.SD_options

            Solver = PPCSteepestDecent(J,grad_J,x0.copy(),PPC
                                       options=SDopt)
            res = Solver.solve()

            result.append(res)
        if len(result)==1:
            return res
        else:
            return result

