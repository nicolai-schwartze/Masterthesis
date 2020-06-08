# -*- coding: utf-8 -*-
"""
Created on Tue Jun  2 18:59:51 2020

@author: Nicolai
"""

import sys
sys.path.append("../../testbed/pde0A/")
import CiPde0A as pde0A
sys.path.append("../../testbed/pde0B/")
import CiPde0B as pde0B

sys.path.append("../../opt_algo")
import OptAlgoDownhillSimplex as oaDS
import OptAlgoMemeticpJADE as oaMpJ

sys.path.append("../../kernels")
import KernelGauss as gk
import KernelGSin as gs

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

if __name__ == "__main__":
    
    # experiment parameter
    replications = 20
    max_fe = 1*10**6
    min_err = 0
    gkernel = gk.KernelGauss()
    gskernel = gs.KernelGSin()
    
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
        
    
    #################################
    #   testbed problem 0A for DS   #
    #################################
    cipde0A = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        ds = oaDS.OptAlgoDownhillSimplex(initialPop, max_fe, min_err)
        cipde0A.append(pde0A.CiPde0A(ds, gkernel, nb2, nc2))
    
    for i in range(replications):
        cipde0A[i].solve()
        pp.saveExpObject(cipde0A[i], "D:/Nicolai/MA_Data/experiment0a/cipde0a_ds_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0a/cipde0a_ds_rep_" + str(i) + ".json" + " -> saved")
    
#    ################################
#    #   testbed problem 0B for DS  #
#    ################################
#    cipde0B = []
#    for i in range(replications):
#        initialPop = np.random.randn(36,18)
#        ds = oaDS.OptAlgoDownhillSimplex(initialPop, max_fe, min_err)
#        cipde0B.append(pde0B.CiPde0B(ds, gskernel, nb2, nc2))
#    
#    for i in range(replications):
#        cipde0B[i].solve()
#        pp.saveExpObject(cipde0B[i], "D:/Nicolai/MA_Data/experiment0a/cipde0b_ds_rep_" + str(i) + ".json")
#        print("D:/Nicolai/MA_Data/experiment0a/cipde0b_ds_rep_" + str(i) + ".json" + " -> saved")
#    
#    
#    ####################################
#    #   testbed problem 0A for mJADE   #
#    ####################################
#    cipde0A = []
#    for i in range(replications):
#        initialPop = np.random.randn(40,20)
#        mpJADE = oaMpJ.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
#        cipde0A.append(pde0A.CiPde0A(mpJADE, gkernel, nb2, nc2))
#    
#    for i in range(replications):
#        cipde0A[i].solve()
#        pp.saveExpObject(cipde0A[i], "D:/Nicolai/MA_Data/experiment0a/cipde0a_mj_rep_" + str(i) + ".json")
#        print("D:/Nicolai/MA_Data/experiment0a/cipde0a_mj_rep_" + str(i) + ".json" + " -> saved")
#    
#    ###################################
#    #   testbed problem 0B for mJADE  #
#    ###################################
#    cipde0B = []
#    for i in range(replications):
#        initialPop = np.random.randn(36,18)
#        mpJADE = oaMpJ.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
#        cipde0B.append(pde0B.CiPde0B(mpJADE, gskernel, nb2, nc2))
#    
#    for i in range(replications):
#        cipde0B[i].solve()
#        pp.saveExpObject(cipde0B[i], "D:/Nicolai/MA_Data/experiment0a/cipde0b_mj_rep_" + str(i) + ".json")
#        print("D:/Nicolai/MA_Data/experiment0a/cipde0b_mj_rep_" + str(i) + ".json" + " -> saved")
#    