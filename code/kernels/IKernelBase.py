# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:51:03 2020

@author: Nicolai
"""

from abc import ABC, abstractmethod


class IKernelBase(ABC):
    """
    Abstract class/interface that any kernel type must fulfill.
    
    Atributes
    ---------
    _kernel_type: string
        descriptive string of the kernel formula
    _kernel_size: int
        number of parameters to be optimised in one kernel
    
    Methods
    -------
    kernel_type(): string
        getter for the attribute kernel_type
    kernel_size(): int
        getter of the attribute kerel_size
    solution(kernels, x): float
        linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0(kernels, x): float
        derivative with respect to x0 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x1(kernels, x): float
        derivative with respect to x1 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0_x0(kernels, x): float
        second derivative with respect to x0 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x1_x1(kernels, x): float
        second derivative with respect to x1 of linear combination of kernels specified in arguemnt, evaluated at x
    solution_x0_x1(kernels, x): float
        second derivative with respect to x0,x1 of linear combination of kernels specified in arguemnt, evaluated at x
    """
    
    def __init__(self):
        self._kernel_type = ""
        self._kernel_size = 0
        
    @property
    @abstractmethod
    def kernel_type(self):
        return self._kernel_type
    
    @property
    @abstractmethod
    def kernel_size(self):
        return self._kernel_size
    
    @abstractmethod
    def solution(self, kernels, x): pass
    
    @abstractmethod
    def solution_x0(self, kernels, x): pass
    
    @abstractmethod
    def solution_x1(self, kernels, x): pass
    
    @abstractmethod
    def solution_x0_x0(self, kernels, x): pass
    
    @abstractmethod
    def solution_x0_x1(self, kernels, x): pass
    
    @abstractmethod
    def solution_x1_x1(self, kernels, x): pass


if __name__ == "__main__":
    pass