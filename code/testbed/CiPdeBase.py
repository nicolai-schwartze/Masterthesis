# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:52:06 2020

@author: Nicolai
"""

from ITestbenchBase import ITestbenchBase
import numpy as np
import scipy as sp

class CiPdeBase(ITestbenchBase):
    
    def __init__(self):
        self._pde_string = ""
        self._exec_time = 0.0
        self._mem_consumption = 0
        
    @property
    def pde_string(self):
        return self._pde_string
    
    @property
    def exec_time(self):
        return self._exec_time
    
    @property
    def mem_consumption(self):
        return self._mem_consumption
    
    def exact(self, x): pass
    
    def approx(self, x): pass

    def normL2(self): pass

    def solve(self): pass


if __name__ == "__main__":
    cipdebase = CiPdeBase()