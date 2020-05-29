# -*- coding: utf-8 -*-
"""
Created on Sun May 24 11:05:18 2020

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
import OptAlgoMemeticJADE as oaMemJade

sys.path.append("../../kernels")
import KernelGauss as gk

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

if __name__ == "__main__":
    
    # experiment parameter
    replications = 20
    max_fe = 5*10**4
    min_err = 10**(-200)
    gkernel = gk.KernelGauss()
    
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
    
    
    ##########################
    #   testbed problem 0A   #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde0A = pde0A.CiPde0A(mJade, gkernel, nb2, nc2)
        cipde0A.solve()
        pp.saveExpObject(cipde0A, "D:/Nicolai/MA_Data/experiment0/cipde0a_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde0a_rep_" + str(i) + ".json" + " -> saved")
    
    ##########################
    #   testbed problem 0B   #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde0B = pde0B.CiPde0B(mJade, gkernel, nb2, nc2)
        cipde0B.solve()
        pp.saveExpObject(cipde0B, "D:/Nicolai/MA_Data/experiment0/cipde0b_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde0b_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 1    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde1 = pde1.CiPde1(mJade, gkernel, nb1, nc1)
        cipde1.solve()
        pp.saveExpObject(cipde1, "D:/Nicolai/MA_Data/experiment0/cipde1_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde1_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 2    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde2 = pde2.CiPde2(mJade, gkernel, nb1, nc1)
        cipde2.solve()
        pp.saveExpObject(cipde2, "D:/Nicolai/MA_Data/experiment0/cipde2_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde2_rep_" + str(i) + ".json" + " -> saved")
    
    ##########################
    #   testbed problem 3    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde3 = pde3.CiPde3(mJade, gkernel, nb1, nc1)
        cipde3.solve()
        pp.saveExpObject(cipde3, "D:/Nicolai/MA_Data/experiment0/cipde3_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde3_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 4    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde4 = pde4.CiPde4(mJade, gkernel, nb1, nc1)
        cipde4.solve()
        pp.saveExpObject(cipde4, "D:/Nicolai/MA_Data/experiment0/cipde4_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde4_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 5    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde5 = pde5.CiPde5(mJade, gkernel, nb1, nc1)
        cipde5.solve()
        pp.saveExpObject(cipde5, "D:/Nicolai/MA_Data/experiment0/cipde5_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde5_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 6    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde6 = pde6.CiPde6(mJade, gkernel, nb1, nc1)
        cipde6.solve()
        pp.saveExpObject(cipde6, "D:/Nicolai/MA_Data/experiment0/cipde6_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde6_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 7    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde7 = pde7.CiPde7(mJade, gkernel, nb1, nc1)
        cipde7.solve()
        pp.saveExpObject(cipde7, "D:/Nicolai/MA_Data/experiment0/cipde7_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde7_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 8    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde8 = pde8.CiPde8(mJade, gkernel, nb1, nc1)
        cipde8.solve()
        pp.saveExpObject(cipde8, "D:/Nicolai/MA_Data/experiment0/cipde8_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde8_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 9    #
    ##########################
    for i in range(replications):
        initialPop = np.random.randn(40,20)
        mJade = oaMemJade.OptAlgoMemeticJADE(initialPop, max_fe, min_err)
        cipde9 = pde9.CiPde9(mJade, gkernel, nb1, nc1)
        cipde9.solve()
        pp.saveExpObject(cipde9, "D:/Nicolai/MA_Data/experiment0/cipde9_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment0/cipde9_rep_" + str(i) + ".json" + " -> saved")
        
        
        
        
    
    
    
    