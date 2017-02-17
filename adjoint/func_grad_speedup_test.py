from my_bfgs.lbfgs import Lbfgs
from my_bfgs.splitLbfgs import SplitLbfgs
from my_bfgs.my_vector import SimpleVector
from penalty import partition_func
from scipy.integrate import trapz
from scipy.optimize import minimize
import numpy as np
import time
import sys

from mpi4py import MPI
from optimalContolProblem import OptimalControlProblem, Problem1
from my_bfgs.mpiVector import MPIVector
from parallelOCP import interval_partition,v_comm_numbers
from ODE_pararealOCP import PararealOCP
from mpiVectorOCP import MpiVectorOCP,simpleMpiVectorOCP,generate_problem,local_u_size

def test_func(N,K,problem,pproblem,name='funcSpeed'):
    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    
        
    if m == 1:
        vals = []
        test_m = 1
        u = np.zeros(N+test_m)+1
        for i in range(K):
            t0 = time.time()
            pproblem.Functional(u,N)#Penalty_Functional(u,N,10,10)
            t1 = time.time()
            vals.append(t1-t0)
        val = min(vals)
        time_saver = open('temp_time.txt','a')
        time_saver.write('%f ' %val)
        time_saver.close()
                    
        #out = open('outputDir/functionalSpeed/'+name+'_'+str(N)+'.txt','w')
        #out.write('seq: %f \n'%val)
        #out.close()
    else:
        
        mu = 10
        u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
        Max_time=[]
        Min_time=[]
        for i in range(K):
            comm.Barrier()
            t0 = time.time()
            pproblem.parallel_penalty_functional(u,N,mu)
            t1 = time.time()
            comm.Barrier()
            loc_time = np.zeros(1)
            loc_time[0] = t1-t0
            if rank == 0:
                time_vec = np.zeros(m)
            else:
                time_vec = None
            loc_size = tuple(np.zeros(m)+1)
            loc_start = tuple(np.linspace(0,m-1,m))
            comm.Gatherv(loc_time,[time_vec,loc_size,loc_start,MPI.DOUBLE])
            if rank == 0:
                Min_time.append(min(time_vec))
                Max_time.append(max(time_vec))
                                
        if rank == 0:
            min_time = min(Min_time)
            max_time = max(Max_time)
            time_saver = open('temp_time.txt','a')
            time_saver.write('%f ' %min_time)
            time_saver.close()
            #out = open('outputDir/functionalSpeed/'+name+'_'+str(N)+'.txt','a')
            #out.write('par %d: %f %f \n'%(m,min_time,max_time))
            #out.close()

def test_grad(N,K,problem,pproblem,name='gradSpeed'):
    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
        
    if m == 1:
        u = np.zeros(N+1)+1
        vals = []
        for i in range(K):
            t0 = time.time()
            pproblem.Gradient(u,N)
            t1 = time.time()
            vals.append(t1-t0)
        val = min(vals)
        time_saver = open('temp_time.txt','a')
        time_saver.write('%f ' %val)
        time_saver.close()
        #out = open('outputDir/gradientSpeed/'+name+'_'+str(N)+'.txt','w')
        #out.write('seq: %f \n'%val)
        #out.close()
    else:
        mu = 10
        u = MPIVector(np.zeros(local_u_size(N+1,m,rank)),comm)
        Max_time=[]
        Min_time=[]
        for i in range(K):                    
            comm.Barrier()
            t0 = time.time()
            pproblem.penalty_grad(u,N,m,mu)
            t1 = time.time()
            comm.Barrier()
            loc_time = np.zeros(1)
            loc_time[0] = t1-t0
            if rank == 0:
                time_vec = np.zeros(m)
            else:
                time_vec = None
            loc_size = tuple(np.zeros(m)+1)
            loc_start = tuple(np.linspace(0,m-1,m))
            comm.Gatherv(loc_time,[time_vec,loc_size,loc_start,MPI.DOUBLE])
            if rank == 0:
                Min_time.append(min(time_vec))
                Max_time.append(max(time_vec))
                                
        if rank == 0:
            #print Min_time
            min_time = min(Min_time)
            max_time = max(Max_time)
            time_saver = open('temp_time.txt','a')
            time_saver.write('%f ' %min_time)
            time_saver.close()
            #out = open('outputDir/gradientSpeed/'+name+'_'+str(N)+'.txt','a')
            #out.write('par %d: %f %f \n'%(m,min_time,max_time))
            #out.close()
def read_temp_time(name,m,seq,rank,solve=False):
    if rank==0:
        temp_time_file = open('temp_time.txt','r')
        time_vals = temp_time_file.readlines()
        vals = []
        for line in time_vals:
            line_list = line.split()
            for l in line_list:
                vals.append(float(l))

        min_val = min(vals)

        if solve:
            if seq:
                info = open('temp_info.txt','r')
                info_vals = info.readlines()
                vals2 = []
                for line in info_vals:
                    line_list = line.split()
                    for l in line_list:
                        vals2.append(int(l))
                seq_string = 'seq: %f %d %d %d %d\n'%(min_val,vals2[0],vals2[1],vals2[2],vals2[3])
            else:
                info = open('temp_info.txt','r')
                info_vals = info.readlines()
                vals2 = []
                for line in info_vals:
                    line_list = line.split()
                    for l in line_list:
                        vals2.append(int(l))
                par_string = 'par %d: %f %d %d %d %d \n'%(m,min_val,vals2[0],vals2[1],vals2[2],vals2[3])
        else:
            seq_string = 'seq: %f \n'%min_val
            par_string = 'par %d: %f \n'%(m,min_val)

        if seq:
            print 
            out = open(name,'w')
            out.write(seq_string)
            out.close()
        else:
            out=open(name,'a')
            out.write(par_string)
            out.close()
        temp_time_file.close()
    return

def test_solve(N,problem,pproblem,name='solveSpeed'):
    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    
    if m == 1:
        opt = {'jtol':1e-10}
        t0 = time.time()
        res = pproblem.solve(N,Lbfgs_options=opt)
        t1=time.time()
        val = t1-t0
        np.save('seq_sol',res.x)
        temp1 = open('temp_time.txt','a')
        temp1.write('%f ' %val)
        temp1.close()
        
        fu,gr=res.counter()

        temp2 = open('temp_info.txt','w')
        temp2.write('%d %d %d %d'%(int(fu),int(gr),res.niter,res.lsiter))
        temp2.close()

    else:
        opt = {'jtol':1e-5}
        mu_list = [N]
        comm.Barrier()
        t0 = time.time()
        res = pproblem.parallel_PPCLBFGSsolve(N,m,mu_list,tol_list=[1e-3,1e-4])#,Lbfgs_options=opt)
        t1 = time.time()
        comm.Barrier()
        res=res[-1]
        loc_time = np.zeros(1)
        loc_time[0] = t1-t0
        if rank == 0:
            time_vec = np.zeros(m)
        else:
            time_vec = None
        loc_size = tuple(np.zeros(m)+1)
        loc_start = tuple(np.linspace(0,m-1,m))
        comm.Gatherv(loc_time,[time_vec,loc_size,loc_start,MPI.DOUBLE])

        if rank==0:

            min_time = min(time_vec)
            
            time_saver = open('temp_time.txt','a')
            time_saver.write('%f ' %min_time)
            time_saver.close()

            fu,gr=res.counter()
            
            info_file = open('temp_info.txt','w')
            info_file.write('%d %d %d %d'%(int(fu),int(gr),res.niter,res.lsiter))
            info_file.close()




def main():
    
    y0 = 1
    yT = 1
    T = 1
    a = 1

    eval_type = sys.argv[4]
    if eval_type == '0':
        name = 'gradientSpeed/gradSpeed'
    else:
        name ='functionalSpeed/funcSpeed'
        
    problem,pproblem=generate_problem(y0,yT,T,a)

    comm = pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    try:
        N = int(sys.argv[1])
    except:
        N = 1000
    try:
        K = int(sys.argv[2])
    except:
        K = 10
    if sys.argv[3]=='0':
        if m==1:
            seq=True
        else:
            seq=False
        name = 'outputDir/'+name+'_'+str(N)+'.txt'
        read_temp_time(name,m,seq,rank)
        return
    if eval_type=='0':
        #test_func(N,K,problem,pproblem)
        test_grad(N,K,problem,pproblem)
    else:
        test_func(N,K,problem,pproblem)
def main2():
    y0 = 1
    yT = 1
    T = 1
    a = 1
    problem,pproblem=generate_problem(y0,yT,T,a)
    try:
        N = int(sys.argv[1])
    except:
        N = 1000
    
    
    comm=pproblem.comm
    m = comm.Get_size()
    rank = comm.Get_rank()
    if sys.argv[3] == '0':
        if m ==1:
            seq = True
        else:
            seq =False
        read_temp_time('outputDir/solveSpeed/sSpeed_'+str(N)+'.txt',m,seq,rank,solve=True)
        return
    test_solve(N,problem,pproblem,name='solveSpeed')
    


if __name__ == '__main__':
    #main()
    main2()
