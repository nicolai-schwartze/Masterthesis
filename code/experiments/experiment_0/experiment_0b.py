# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 13:52:23 2020

@author: Nicolai
"""

import sys
sys.path.append("../../testbed/pde0A/")
import CiPde0A as pde0A

sys.path.append("../../opt_algo")
import OptAlgoMemeticJADE as oaMJ

sys.path.append("../../kernels")
import KernelGauss as gk

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

if __name__ == "__main__":
    
    # experiment parameter
    replications = 1
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
        
    
    ####################################
    #   testbed problem 0A for mpJADE  #
    ####################################
    cipde0A = []
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJADE = oaMJ.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde0A.append(pde0A.CiPde0A(mJADE, gakernel, nb2, nc2))
    
    for i in range(replications):
        cipde0A[i].solve()
        pp.saveExpObject(cipde0A[i], "../../cipde0a_mj_rep_" + str(i) + ".json")
        print("../../cipde0a_mj_rep_" + str(i) + ".json" + " -> saved")
    