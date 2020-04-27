# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 20:52:06 2020

@author: Nicolai
"""

from ITestbenchBase import ITestbenchBase
import numpy as np
import scipy.integrate as integrate
import time
import psutil

class CiPdeBase(ITestbenchBase):
    """
    Abstract class that implements the ITestbenchBase interface. 
    It further provides functionality that is used by the CI solver. 
    
    Attributes
    ----------
    opt_algo: IOptAlgoBase
        implementation of an IOptAlgoBase for any optimisation algorithm
    kernel: IKernelBase
        KernelGauss or any other implementation of the IKernelBase
    pop_history: list
        population at every iteration during the optimisation
    fit_history: list
        fitness value at every iteration during the optimisation
    cr_history: list
        crossover probability at every iteration
    f_history: list
        scalar factor at every iteration
    _nb: list
        list of evaluation point (tupels) on the boundary
    _nc: list
        list of inner evaluation points stored as tupels
    _weight_reference: float
        denominator of weighting factor stored to only compute once
    _lx: float 
        lower x value at of the domain
    _ux: float
        upper x limit of the domain
    _xi: list
        weighting factor for inner collocation points
    _phi: list
        weighting factor for boundary point
        
    Methods
    -------
    pde_string(): string
        getter for returning the _pde_string that holds a short description of the problem
    exec_time(): float
        getter for returning the execution time taken for solving the probem
    mem_consumption(): int
        getter for returning the memory consumption of the solver
    exact(x): flaot
        takes a numpy array, returns the function value of the exact solution 
    approx(x): float
        takes a numpy array,returns the function value of the approximate solution
    normL2(): float
        returns the distance between the exact and the approximate solution
    solve(): None
        not implemented, must be overridden by the child class that is specific to a pde
    nc_weight(xi): float
        calculates the weighting factor for one specific xi from nc
    fitness_func(kernels): float
        objective function passed to the optimisation algorithm, not implemented, must be overridden by the child class that is specific to a pde
    _ly(x): float
        lower y boundary of the domain
    _uy(x): float
        upper y boundary of the domain
    
    """
    def __init__(self, opt_algo, kernel, nb, nc):
        self.opt_algo = opt_algo
        self.kernel = kernel
        self._pde_string = ""
        self._exec_time = 0.0
        self._mem_consumption = 0
        self.pop_history = []
        self.fit_history = []
        self.cr_history = []
        self.f_history = []
        self.sol_kernel = np.array([])
        self._nc = nc
        self._nb = nb
        # inner weightin factor reference term
        temp_denominator = []
        for xk in nc:
            temp_xk_xj = []
            for xj in nb:
                temp_xk_xj.append(np.linalg.norm(np.array([xk[0], xk[1]])-np.array([xj[0], xj[1]])))
            temp_denominator.append(min(temp_xk_xj))
        self._weight_reference = max(temp_denominator)
        self._xi = []
        self._phi = []
        self._kappa = 0
        self._lx = None
        self._ux = None
        self._ly = None
        self._uy = None
        
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
    
    def approx(self, x): 
        try: 
            return self.kernel.solution(self.sol_kernel, x)
        except ValueError: 
            print("self.sol_kernel is empty, try to call solve() first")

    def normL2(self):
        difference_func = lambda x,y: \
        (self.approx(np.array([x,y])) - self.exact(np.array([x,y]))) * \
        (self.approx(np.array([x,y])) - self.exact(np.array([x,y])))
        return np.sqrt(integrate.dblquad(difference_func, self._lx, self._ux, self._ly, self._uy)[0])

    def fitness_func(self, kernels): pass

    def nc_weight(self, xi):
        temp_xi_xj = []
        for xj in self._nb:
            temp_xi_xj.append(np.linalg.norm(np.array([xi[0], xi[1]])-np.array([xj[0], xj[1]])))
        
        return (1 + self._kappa*(1-(min(temp_xi_xj)/self._weight_reference)))/(1+self._kappa)

    def solve(self): 
        # start memory measurement
        process = psutil.Process()
        memstart = process.memory_info().rss
        
        # start timer 
        t_start = time.time()
        
        # call optimisation algorithm
        self.pop_history, self.fit_history, self.f_history, self.cr_history  = \
        self.opt_algo.opt(self.fitness_func)
        
        # save found solution to self.sol_kernel
        bestIndex = np.argmin(self.fit_history[-1])
        kernel = self.pop_history[-1][bestIndex]
        dimension = kernel.shape[0]
        numberOfKernels = dimension/self.kernel.kernel_size
        # candidate solution coding: 
        # [wi, yi, c1i, c2i] where i is the number of kernels which is adaptive
        kernel = kernel.reshape((int(numberOfKernels), self.kernel.kernel_size))
        self.sol_kernel = kernel
        
        # stop timer 
        self._exec_time = time.time() - t_start
        
        # stop memory measurement
        process = psutil.Process()
        memstop = process.memory_info().rss - memstart
        
        # subtract lists because they are not needed? 
        # -> ask Steffen
        
        self._mem_consumption = memstop


if __name__ == "__main__":
    pass
#    cipdebase = CiPdeBase(None, None)
    