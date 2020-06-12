# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:52:23 2020

@author: Nicolai
"""

import sys
sys.path.append("../../testbed/pde0A/")
import CiPde0A as pde0A
sys.path.append("../../testbed/pde0B/")
import CiPde0B as pde0B
sys.path.append("../../testbed/pde1/")
import CiPde1 as pde1
sys.path.append("../../testbed/pde2/")
import CiPde2 as pde2
sys.path.append("../../testbed/pde3/")
import CiPde3 as pde3
sys.path.append("../../testbed/pde4/")
import CiPde4 as pde4
sys.path.append("../../testbed/pde5/")
import CiPde5 as pde5
sys.path.append("../../testbed/pde6/")
import CiPde6 as pde6
sys.path.append("../../testbed/pde7/")
import CiPde7 as pde7
sys.path.append("../../testbed/pde8/")
import CiPde8 as pde8
sys.path.append("../../testbed/pde9/")
import CiPde9 as pde9

sys.path.append("../../opt_algo")
import OptAlgoMemeticJADE as oaMJ

sys.path.append("../../kernels")
import KernelGauss as gk

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

from multiprocessing import Pool

def solveObj(obj_list, i):
    obj_list[i].solve()
    return obj_list[i]

if __name__ == "__main__":
    
    # experiment parameter
    replications = 20
    max_fe = 1*10**6
    min_err = 0
    gakernel = gk.KernelGauss()
    
    # collocation points for 0A and 0B
    nc2 = []
    omega = np.arange(-1.6, 2.0, 0.4)
    for x0 in omega:
        for x1 in omega:
            nc2.append((x0, x1))
        
    # boundary points for 0A and 0B
    nb2 = []
    nby = np.hstack((-2*np.ones(10), np.arange(-2.0, 2.0, 0.4), 2*np.ones(10), np.arange(2.0, -2.0, -0.4)))
    nbx = np.hstack((np.arange(-2.0, 2.0, 0.4), 2*np.ones(10), np.arange(2.0, -2.0, -0.4), -2*np.ones(10)))
    for i in range(len(nby)):
        nb2.append((nbx[i], nby[i]))
        
    # collocation points
    nc1 = []
    omega = np.arange(0.1, 1.0, 0.1)
    for x0 in omega:
        for x1 in omega:
            nc1.append((x0, x1))
        
    # boundary points
    nb1 = []
    nby = np.hstack((np.zeros(10), np.arange(0.0, 1.0, 0.1), np.ones(10), np.arange(1.0, 0.0, -0.1)))
    nbx = np.hstack((np.arange(0.0, 1.0, 0.1), np.ones(10), np.arange(1.0, 0.0, -0.1), np.zeros(10)))
    for i in range(40):
        nb1.append((nbx[i], nby[i]))
    
    
    
    
    
#    ####################################
#    #   testbed problem 0A for mJADE   #
#    ####################################
#    cipde0A = []
#    for i in range(replications):
#        initialPop = np.random.randn(40,20)
#        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
#        cipde0A.append(pde0A.CiPde0A(mJADE, gakernel, nb2, nc2))
#        
#    pool = Pool(processes= 20)
#    parallelResults = []
#    for i in range(replications):
#        parallelResults.append(pool.apply_async(solveObj, args=(cipde0A, i)))
#    
#    for i in range(replications):
#        obj = parallelResults[i].get()
#        pp.saveExpObject(obj, "../../cipde0a_mj_rep_" + str(i) + ".json")
#        print("../../cipde0a_mj_rep_" + str(i) + ".json" + " -> saved")
#    
#    pool.close()
#    pool.join()
    
    
    
    
    
    ####################################
    #   testbed problem 0B for mJADE   #
    ####################################
    cipde0B = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde0B.append(pde0B.CiPde0B(mJADE, gakernel, nb2, nc2))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde0B, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde0b_mj_rep_" + str(i) + ".json")
        print("../../cipde0b_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 1 for mJADE   #
    ####################################
    cipde1 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde1.append(pde1.CiPde1(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde1, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde1_mj_rep_" + str(i) + ".json")
        print("../../cipde1_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 2 for mJADE   #
    ####################################
    cipde2 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde2.append(pde2.CiPde2(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde2, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde2_mj_rep_" + str(i) + ".json")
        print("../../cipde2_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 3 for mJADE   #
    ####################################
    cipde3 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde3.append(pde3.CiPde3(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde3, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde3_mj_rep_" + str(i) + ".json")
        print("../../cipde3_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 4 for mJADE   #
    ####################################
    cipde4 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde4.append(pde4.CiPde4(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde4, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde4_mj_rep_" + str(i) + ".json")
        print("../../cipde4_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 5 for mJADE   #
    ####################################
    cipde5 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde5.append(pde5.CiPde5(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde5, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde5_mj_rep_" + str(i) + ".json")
        print("../../cipde5_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 6 for mJADE   #
    ####################################
    cipde6 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde6.append(pde6.CiPde6(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde6, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde6_mj_rep_" + str(i) + ".json")
        print("../../cipde6_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 7 for mJADE   #
    ####################################
    cipde7 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde7.append(pde7.CiPde7(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde7, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde7_mj_rep_" + str(i) + ".json")
        print("../../cipde7_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 8 for mJADE   #
    ####################################
    cipde8 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde8.append(pde8.CiPde8(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde8, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde8_mj_rep_" + str(i) + ".json")
        print("../../cipde8_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    
    
    
    
    ####################################
    #    testbed problem 9 for mJADE   #
    ####################################
    cipde9 = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde9.append(pde9.CiPde9(mJADE, gakernel, nb1, nc1))
        
    pool = Pool(processes= 20)
    parallelResults = []
    for i in range(replications):
        parallelResults.append(pool.apply_async(solveObj, args=(cipde9, i)))
    
    for i in range(replications):
        obj = parallelResults[i].get()
        pp.saveExpObject(obj, "../../cipde9_mj_rep_" + str(i) + ".json")
        print("../../cipde9_mj_rep_" + str(i) + ".json" + " -> saved")
    
    pool.close()
    pool.join()
    
    