# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 23:51:03 2020

@author: Nicolai
"""

from abc import ABC, abstractmethod


class IKernelBase(ABC):
    
    def __init__(self):
        self.__kernel_type = ""
        
    @property
    @abstractmethod
    def kernel_type(self):
        return self.__kernel_type
    
    @abstractmethod
    def solution(self, x): pass
    
    @abstractmethod
    def solution_x0(self, x): pass
    
    @abstractmethod
    def solution_x1(self, x): pass
    
    @abstractmethod
    def solution_x0_x0(self, x): pass
    
    @abstractmethod
    def solution_x0_x1(self, x): pass
    
    @abstractmethod
    def solution_x1_x1(self, x): pass


if __name__ == "__main__":
    pass