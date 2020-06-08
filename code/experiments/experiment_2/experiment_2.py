# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 19:31:42 2020

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
import OptAlgoMemeticpJADE as oaMempJade

sys.path.append("../../kernels")
import KernelGSin as gsk

import numpy as np

sys.path.append("../../post_proc/")
import post_proc as pp

if __name__ == "__main__":
    
    # experiment parameter
    replications = 20
    max_fe = 1*10**4
    min_err = 0
    gskernel = gsk.KernelGSin()
    
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
    cipde0A = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde0A.append(pde0A.CiPde0A(mpJade, gskernel, nb2, nc2))
        
    initialPop = np.random.randn(36,18)
    mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
    remove_outlier = pde0A.CiPde0A(mpJade, gskernel, nb2, nc2)
    remove_outlier.solve()
    
    for i in range(replications):
        cipde0A[i].solve()
        pp.saveExpObject(cipde0A[i], "D:/Nicolai/MA_Data/experiment2/cipde0a_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde0a_rep_" + str(i) + ".json" + " -> saved")
    
    ##########################
    #   testbed problem 0B   #
    ##########################
    cipde0B = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde0B.append(pde0B.CiPde0B(mpJade, gskernel, nb2, nc2))
        
    for i in range(replications):
        cipde0B[i].solve()
        pp.saveExpObject(cipde0B[i], "D:/Nicolai/MA_Data/experiment2/cipde0b_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde0b_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 1    #
    ##########################
    cipde1 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde1.append(pde1.CiPde1(mpJade, gskernel, nb1, nc1))
    
    for i in range(replications):
        cipde1[i].solve()
        pp.saveExpObject(cipde1[i], "D:/Nicolai/MA_Data/experiment2/cipde1_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde1_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 2    #
    ##########################
    cipde2 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde2.append(pde2.CiPde2(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde2[i].solve()
        pp.saveExpObject(cipde2[i], "D:/Nicolai/MA_Data/experiment2/cipde2_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde2_rep_" + str(i) + ".json" + " -> saved")
    
    ##########################
    #   testbed problem 3    #
    ##########################
    cipde3 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde3.append(pde3.CiPde3(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde3[i].solve()
        pp.saveExpObject(cipde3[i], "D:/Nicolai/MA_Data/experiment2/cipde3_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde3_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 4    #
    ##########################
    cipde4 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde4.append(pde4.CiPde4(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde4[i].solve()
        pp.saveExpObject(cipde4[i], "D:/Nicolai/MA_Data/experiment2/cipde4_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde4_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 5    #
    ##########################
    cipde5 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde5.append(pde5.CiPde5(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde5[i].solve()
        pp.saveExpObject(cipde5[i], "D:/Nicolai/MA_Data/experiment2/cipde5_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde5_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 6    #
    ##########################
    cipde6 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde6.append(pde6.CiPde6(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde6[i].solve()
        pp.saveExpObject(cipde6[i], "D:/Nicolai/MA_Data/experiment2/cipde6_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde6_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 7    #
    ##########################
    cipde7 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde7.append(pde7.CiPde7(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde7[i].solve()
        pp.saveExpObject(cipde7[i], "D:/Nicolai/MA_Data/experiment2/cipde7_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde7_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 8    #
    ##########################
    cipde8 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde8.append(pde8.CiPde8(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde8[i].solve()
        pp.saveExpObject(cipde8[i], "D:/Nicolai/MA_Data/experiment2/cipde8_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde8_rep_" + str(i) + ".json" + " -> saved")
        
    ##########################
    #   testbed problem 9    #
    ##########################
    cipde9 = []
    for i in range(replications):
        initialPop = np.random.randn(36,18)
        mpJade = oaMempJade.OptAlgoMemeticpJADE(initialPop, max_fe, min_err)
        cipde9.append(pde9.CiPde9(mpJade, gskernel, nb1, nc1))
        
    for i in range(replications):
        cipde9[i].solve()
        pp.saveExpObject(cipde9[i], "D:/Nicolai/MA_Data/experiment2/cipde9_rep_" + str(i) + ".json")
        print("D:/Nicolai/MA_Data/experiment2/cipde9_rep_" + str(i) + ".json" + " -> saved")
        
        
        
        
    
    
    
    