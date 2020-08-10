# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 11:31:44 2020

@author: Nicolai
"""


import sys 
import pickle

# needed to import FemPdeBase in subfiles
sys.path.append("../../testbed")

sys.path.append("../../testbed/pde3/")
import FemPde3 as pde3


replication = 20

if __name__ == "__main__": 
    
    
    fempde3_5  = []
    fempde3_5_result  = []
    
    for r in range(replication+1):
        fempde3_5.append(pde3.FemPde3(False, max_ndof=5))
        
    # solving fem pde 3
    fempde3_5[0].solve()
    for r in range(1,replication+1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3_5[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3_5[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3_5[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3_5[r].normL2()
        print("dist to analyt: " + str(dist))
        fempde3_5_result.append([time, mem, dist])
        print("\n")
        
    with open("fem_pde3_5dof.p", "wb") as f:
        pickle.dump(fempde3_5_result, f)
        
    fempde3_50  = []
    fempde3_50_result  = []
    
    for r in range(replication+1):
        fempde3_50.append(pde3.FemPde3(False, max_ndof=50))
        
    # solving fem pde 3
    fempde3_50[0].solve()
    for r in range(1,replication+1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3_50[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3_50[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3_50[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3_50[r].normL2()
        print("dist to analyt: " + str(dist))
        fempde3_50_result.append([time, mem, dist])
        print("\n")
        
    with open("fem_pde3_50dof.p", "wb") as f:
        pickle.dump(fempde3_50_result, f)
        
    fempde3_500  = []
    fempde3_500_result  = []
    
    for r in range(replication+1):
        fempde3_500.append(pde3.FemPde3(False, max_ndof=500))
        
    # solving fem pde 3
    fempde3_500[0].solve()
    for r in range(1,replication+1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3_500[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3_500[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3_500[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3_500[r].normL2()
        print("dist to analyt: " + str(dist))
        fempde3_500_result.append([time, mem, dist])
        print("\n")
        
    with open("fem_pde3_500dof.p", "wb") as f:
        pickle.dump(fempde3_500_result, f)
        
    fempde3_5000  = []
    fempde3_5000_result  = []
    
    for r in range(replication+1):
        fempde3_5000.append(pde3.FemPde3(False, max_ndof=5000))
        
    # solving fem pde 3
    fempde3_5000[0].solve()
    for r in range(1,replication+1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3_5000[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3_5000[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3_5000[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3_5000[r].normL2()
        print("dist to analyt: " + str(dist))
        fempde3_5000_result.append([time, mem, dist])
        print("\n")
        
    with open("fem_pde3_5000dof.p", "wb") as f:
        pickle.dump(fempde3_5000_result, f)
        
    fempde3_50000  = []
    fempde3_50000_result  = []
    
    for r in range(replication+1):
        fempde3_50000.append(pde3.FemPde3(False, max_ndof=50000))
        
    # solving fem pde 3
    fempde3_50000[0].solve()
    for r in range(1,replication+1):
        print("solving pde 3, replication " + str(r))
        print(30*"-")
        fempde3_50000[r].solve()
        print("\n")
        print("results: ")
        print(30*"-")
        time = fempde3_50000[r].exec_time
        print("execution time: " + str(time))
        mem  = fempde3_50000[r].mem_consumption
        print("memory usage  : " + str(mem))
        dist = fempde3_50000[r].normL2()
        print("dist to analyt: " + str(dist))
        fempde3_50000_result.append([time, mem, dist])
        print("\n")
        
    with open("fem_pde3_50000dof.p", "wb") as f:
        pickle.dump(fempde3_50000_result, f)