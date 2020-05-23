# -*- coding: utf-8 -*-
"""
Created on Tue May 12 10:06:56 2020

@author: Nicolai
"""

import sys 
import pickle 

# needed to import FemPdeBase in subfiles
sys.path.append("../../testbed")

sys.path.append("../../testbed/pde0A/")
import FemPde0A as pde0A
sys.path.append("../../testbed/pde0B/")
import FemPde0B as pde0B
sys.path.append("../../testbed/pde1/")
import FemPde1 as pde1
sys.path.append("../../testbed/pde2/")
import FemPde2 as pde2
sys.path.append("../../testbed/pde3/")
import FemPde3 as pde3
sys.path.append("../../testbed/pde4/")
import FemPde4 as pde4
sys.path.append("../../testbed/pde5/")
import FemPde5 as pde5
sys.path.append("../../testbed/pde6/")
import FemPde6 as pde6
sys.path.append("../../testbed/pde7/")
import FemPde7 as pde7
sys.path.append("../../testbed/pde8/")
import FemPde8 as pde8
sys.path.append("../../testbed/pde9/")
import FemPde9 as pde9

replication = 20

if __name__ == "__main__": 
    
    print("starting FEM base experiment")
    
    fempde0A = []
    fempde0B = []
    fempde1  = []
    fempde2  = []
    fempde3  = []
    fempde4  = []
    fempde5  = []
    fempde6  = []
    fempde7  = []
    fempde8  = []
    fempde9  = []
    fempde0A_r = []
    fempde0B_r = []
    fempde1_r  = []
    fempde2_r  = []
    fempde3_r  = []
    fempde4_r  = []
    fempde5_r  = []
    fempde6_r  = []
    fempde7_r  = []
    fempde8_r  = []
    fempde9_r  = []
    
    
    for r in range(replication + 1):
        fempde0A.append(pde0A.FemPde0A(False))
        fempde0B.append(pde0B.FemPde0B(False))
        fempde1.append(pde1.FemPde1(False))
        fempde2.append(pde2.FemPde2(False))
        fempde3.append(pde3.FemPde3(False))
        fempde4.append(pde4.FemPde4(False))
        fempde5.append(pde5.FemPde5(False))
        fempde6.append(pde6.FemPde6(False))
        fempde7.append(pde7.FemPde7(False))
        fempde8.append(pde8.FemPde8(False))
        fempde9.append(pde9.FemPde9(False))
        
    # solving fem pde 0A
    fempde0A[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 0A, replication " + str(r))
        print(30*"-")
        fempde0A[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde0A[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde0A[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde0A[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde0A_r.append(result)
        print("\n")
        
    with open("fem_pde0A_baseline_data.p", "wb") as f:
        pickle.dump(fempde0A_r, f)
        
    # solving fem pde 0B
    fempde0B[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 0B, replication " + str(r))
        print(30*"-")
        fempde0B[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde0B[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde0B[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde0B[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde0B_r.append(result)
        print("\n")
        
    with open("fem_pde0B_baseline_data.p", "wb") as f:
        pickle.dump(fempde0B_r, f)
        
    # solving fem pde 1
    fempde1[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 1, replication " + str(r))
        print(30*"-")
        fempde1[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde1[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde1[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde1[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde1_r.append(result)
        print("\n")
        
    with open("fem_pde1_baseline_data.p", "wb") as f:
        pickle.dump(fempde1_r, f)
        
    # solving fem pde 2
    fempde2[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 2, replication " + str(r))
        print(30*"-")
        fempde2[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde2[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde2[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde2[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde2_r.append(result)
        print("\n")
        
    with open("fem_pde2_baseline_data.p", "wb") as f:
        pickle.dump(fempde2_r, f)
        
    # solving fem pde 3
    fempde3[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde3_r.append(result)
        print("\n")
        
    with open("fem_pde3_baseline_data.p", "wb") as f:
        pickle.dump(fempde3_r, f)
        
    # solving fem pde 4
    fempde4[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 4, replication " + str(r))
        print(30*"-")
        fempde4[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde4[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde4[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde4[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde4_r.append(result)
        print("\n")
        
    with open("fem_pde4_baseline_data.p", "wb") as f:
        pickle.dump(fempde4_r, f)
        
    # solving fem pde 5
    fempde5[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 5, replication " + str(r))
        print(30*"-")
        fempde5[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde5[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde5[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde5[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde5_r.append(result)
        print("\n")
        
    with open("fem_pde5_baseline_data.p", "wb") as f:
        pickle.dump(fempde5_r, f)
    
    # solving fem pde 6
    fempde6[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 6, replication " + str(r))
        print(30*"-")
        fempde6[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde6[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde6[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde6[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde6_r.append(result)
        print("\n")
        
    with open("fem_pde6_baseline_data.p", "wb") as f:
        pickle.dump(fempde6_r, f)
    
    # solving fem pde 7
    fempde7[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 7, replication " + str(r))
        print(30*"-")
        fempde7[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde7[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde7[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde7[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde7_r.append(result)
        print("\n")
        
    with open("fem_pde7_baseline_data.p", "wb") as f:
        pickle.dump(fempde7_r, f)
    
    # solving fem pde 8
    fempde8[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 8, replication " + str(r))
        print(30*"-")
        fempde8[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde8[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde8[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde8[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde8_r.append(result)
        print("\n")
        
    with open("fem_pde8_baseline_data.p", "wb") as f:
        pickle.dump(fempde8_r, f)
    
    # solving fem pde 9
    fempde9[0].solve()
    for r in range(1, replication + 1):
        print("solving pde 9, replication " + str(r))
        print(30*"-")
        fempde9[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde9[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde9[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde9[r].normL2()
        print("dist to analyt: " + str(dist))
        result = [time, mem, dist]
        fempde9_r.append(result)
        print("\n")
        
    with open("fem_pde9_baseline_data.p", "wb") as f:
        pickle.dump(fempde9_r, f)
        

    print("finished experiment")
    
    
    
    