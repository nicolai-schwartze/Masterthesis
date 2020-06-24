# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 20:11:02 2020

@author: Nicolai
----------------
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
import OptAlgoMemeticpJADEadaptive as oapMJa

sys.path.append("../../kernels")
import KernelGauss as gk

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

if __name__ == "__main__":
    
    # experiment parameter
    replications = 3
    max_fe = 1*10**4
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
    
    
    
    
    
    ####################################
    #  testbed problem 0A for mpJADEa  #
    ####################################
    cipde0A = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde0A.append(pde0A.CiPde0A(mpJADEa, gakernel, nb2, nc2))

    for i in range(replications):
        cipde0A[i].solve()
        pp.saveExpObject(cipde0A[i], "../../cipde0a_mpja_rep_" + str(i) + ".json")
        print("../../cipde0a_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #  testbed problem 0B for mpJADEa  #
    ####################################
    cipde0B = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde0B.append(pde0B.CiPde0B(mpJADEa, gakernel, nb2, nc2))
        
    for i in range(replications):
        cipde0B[i].solve()
        pp.saveExpObject(cipde0B[i], "../../cipde0b_mpja_rep_" + str(i) + ".json")
        print("../../cipde0b_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 1 for mpJADEa  #
    ####################################
    cipde1 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde1.append(pde1.CiPde1(mpJADEa, gakernel, nb1, nc1))
        
    for i in range(replications):
        cipde1[i].solve()
        pp.saveExpObject(cipde1[i], "../../cipde1_mpja_rep_" + str(i) + ".json")
        print("../../cipde1_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 2 for mpJADEa  #
    ####################################
    cipde2 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde2.append(pde2.CiPde2(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde2[i].solve()
        pp.saveExpObject(cipde2[i], "../../cipde2_mpja_rep_" + str(i) + ".json")
        print("../../cipde2_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 3 for mpJADEa  #
    ####################################
    cipde3 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde3.append(pde3.CiPde3(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde3[i].solve()
        pp.saveExpObject(cipde3[i], "../../cipde3_mpja_rep_" + str(i) + ".json")
        print("../../cipde3_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 4 for mpJADEa  #
    ####################################
    cipde4 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde4.append(pde4.CiPde4(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde4[i].solve()
        pp.saveExpObject(cipde4[i], "../../cipde4_mpja_rep_" + str(i) + ".json")
        print("../../cipde4_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 5 for mpJADEa  #
    ####################################
    cipde5 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde5.append(pde5.CiPde5(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde5[i].solve()
        pp.saveExpObject(cipde5[i], "../../cipde5_mpja_rep_" + str(i) + ".json")
        print("../../cipde5_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 6 for mpJADEa  #
    ####################################
    cipde6 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde6.append(pde6.CiPde6(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde6[i].solve()
        pp.saveExpObject(cipde6[i], "../../cipde6_mpja_rep_" + str(i) + ".json")
        print("../../cipde6_mpja_rep_" + str(i) + ".json" + " -> saved")

    
    
    
    
    
    ####################################
    #   testbed problem 7 for mpJADEa  #
    ####################################
    cipde7 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde7.append(pde7.CiPde7(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde7[i].solve()
        pp.saveExpObject(cipde7[i], "../../cipde7_mpja_rep_" + str(i) + ".json")
        print("../../cipde7_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 8 for mpJADEa  #
    ####################################
    cipde8 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde8.append(pde8.CiPde8(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde8[i].solve()
        pp.saveExpObject(cipde8[i], "../../cipde8_mpja_rep_" + str(i) + ".json")
        print("../../cipde8_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    
    
    
    ####################################
    #   testbed problem 9 for mpJADEa  #
    ####################################
    cipde9 = []
    for i in range(replications):
        initialPop = np.random.randn(8,4)
        mpJADEa = oapMJa.OptAlgoMemeticpJADEadaptive(initialPop, max_fe, min_err)
        cipde9.append(pde9.CiPde9(mpJADEa, gakernel, nb1, nc1))

    for i in range(replications):
        cipde9[i].solve()
        pp.saveExpObject(cipde9[i], "../../cipde9_mpja_rep_" + str(i) + ".json")
        print("../../cipde9_mpja_rep_" + str(i) + ".json" + " -> saved")
    
    
    