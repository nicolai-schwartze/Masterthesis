# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 15:26:53 2020

@author: Nicolai
"""

from abc import ABC, abstractmethod

class ITestbenchBase(ABC):
    """
    Abstract class that provides the basic attributs and methods 
    that any solver of this testbed must implement. 
    
    Attributes
    ----------
    _pde_string: string
        protected: short descriptive string of the pde
    
    _exec_time: float
        protected: execution time to solve in seconds
    
    _mem_consumption: int
        protected: memory consumption in bytes
    
    Methods
    -------
    pde_string()
        getter for returning the _pde_string that holds a short description of the problem
    exec_time()
        getter for returning the execution time taken for solving the probem
    mem_consumption()
        getter for returning the memory consumption of the solver
    exact(x)
        takes a numpy array, returns the function value of the exact solution 
    approx(x)
        takes a numpy array, wrapper for the approximate soltuion, returns the function value of the approximate solution
    normL2()
        returns the distance between the exact and the approximate solution
    solve()
        solves the pde problem, this function is specific to every problem and must be overridden by every pde implementation
    
    """
    def __init__(self):
        self._pde_string = ""
        self._exec_time = 0.0
        self._mem_consumption = 0
        
    @property
    @abstractmethod
    def pde_string(self):
        return self._pde_string
    
    @property
    @abstractmethod
    def exec_time(self):
        return self._exec_time
    
    @property
    @abstractmethod
    def mem_consumption(self):
        return self._mem_consumption
    
    @abstractmethod
    def exact(self, x): pass
    
    @abstractmethod
    def approx(self, x): pass

    @abstractmethod
    def normL2(self): pass

    @abstractmethod
    def solve(self): pass
    

if __name__ == "__main__":
    pass