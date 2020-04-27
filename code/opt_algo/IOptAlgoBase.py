# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 08:09:07 2020

@author: Nicolai
"""

from abc import ABC, abstractmethod

class IOptAlgoBase(ABC):
    """
    Abstract class that provides the basic attributs and methods 
    for the OptAlgo class. This is a level of abstraction, so the optimisation
    algorithm can easily be swichted between experiments.
    
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
    def __init__(self, init_guess, max_iter, min_err):
        self._init_guess = init_guess
        self._max_iter = max_iter
        self._min_err = min_err
    
    @property
    @abstractmethod
    def init_guess(self):
        return self._init_guess
    
    @property
    @abstractmethod
    def max_iter(self):
        return self._max_iter
    
    @property
    @abstractmethod
    def min_err(self):
        return self._min_err
    
    @abstractmethod
    def opt(self, function):
        pass
        
    
    
if __name__ == "__main__":
    pass