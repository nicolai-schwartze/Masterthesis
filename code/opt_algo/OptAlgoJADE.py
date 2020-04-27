# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:04:44 2020

@author: Nicolai
"""

from IOptAlgoBase import IOptAlgoBase
from JADE import JADE

class OptAlgoJADE(IOptAlgoBase):
    """
    Implementation of the IOptAlgoBase interface. 
    It provides a wrapper for the JADE optimisation algorithm
    
    Attributes
    ----------
    _init_guess: np.array
        protected: initial guess for the optimisation (population or position)
    
    _max_iter: int
        protected: stopping condition of maximum number of iteration
    
    _min_err: float
        protected: stopping condition for minimum error (function value) to reach
    
    Methods
    -------
    init_guess()
        getter for returning the _init_guess

    max_iter()
        getter for returning the _max_iter
    
    min_err()
        getter for returning the _min_err
        
    opt(function)
        returns a history tuple of 4 elements
        
        tupel[0] - popDynamic
            
        tupel[1] - FEDynamic
            
        tupel[2] - FDynamic
            
        tupel[3] - CRDynamic
    
    """
    
    def __init__(self, init_guess, max_gen, min_err):
        super().__init__(init_guess, max_gen, min_err)
    
    @property
    def init_guess(self):
        return self._init_guess
    
    @property
    def max_iter(self):
        return self._max_iter
    
    @property
    def min_err(self):
        return self._min_err

    def opt(self, function):
        p = 0.3
        c = 0.5
        return JADE(self.init_guess, p, c, function, self.min_err, self.max_iter)
        

if __name__ == "__main__": 
    
    import numpy as np
    
    initialPop = 100*np.random.rand(4,2)
    max_iter = 10**3
    min_err = 10**(-20)
    jade = OptAlgoJADE(initialPop, max_iter, min_err)
    print(jade.max_iter)
    print(jade.min_err)
    print(jade.init_guess)
    
    def sphere(x):
        return np.dot(x,x)
    
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = jade.opt(sphere)
    import matplotlib.pyplot as plt
    
    plt.semilogy(FEDynamic)
    