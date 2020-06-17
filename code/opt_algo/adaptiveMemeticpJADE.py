# -*- coding: utf-8 -*-
"""
Created on Mon Jun 05 19:12:17 2020

@author: Nicolai
"""

import sys
sys.path.append("../differential_evolution")
from pJADEadaptive import pJADEadaptive
import numpy as np
import scipy as sc
import testFunctions as tf

def memeticpJADEadaptive(population, function, minError, maxFeval): 
    '''
    implementation of an adaptive kernel memetic pJADE: \n
    maxFeval-2*dim of the function evaluation are spend on pJADE
    2*dim of the function evaluation is used to perform a downhill simplex
    internal parameters of pJADE are set to p=0.3 and c=0.5
    the maximum number of parallel processes is set to 5
    
    
    Parameters
    ----------
    population: numpy array
        2D numpy array where lines are candidates and colums is the dimension
    function: function
        fitness function that is optimised
    minError: float 
        stopping condition on function value
    maxFeval: int
        stopping condition on max number of function evaluation
    
    Returns
    -------
    history: tuple
        tupel[0] - popDynamic
        
        tupel[1] - FEDynamic
        
        tupel[2] - FDynamic
        
        tupel[3] - CRDynamic
        
    Examples
    --------
    >>> import numpy as np
    >>> def sphere(x):
            return np.dot(x,x)
    >>> minError = -1*np.inf
    >>> maxGen = 10**3
    >>> population = 100*np.random.rand(50,2)
    >>> (popDynamic, FEDynamic, FDynamic, CRDynamic) = 
        JADE(population, sphere, minError, maxGen)
    
    '''
    psize, dim = population.shape
    p = 0.3
    c = 0.5
    
    print("starting with dimension: " + str(dim))
    
    popDynamic, FEDynamic, FDynamic, CRDynamic = pJADEadaptive(population, p, c, function, minError, maxFeval-2*dim)
    print("finished JADE with " str(maxFeval-2*dim))
    print("="*45)
    print("start direct search with downhill simplex")
    bestIndex = np.argmin(FEDynamic[-1])
    bestSolution = popDynamic[-1][bestIndex]
    _, _, _, _, _, pop  = sc.optimize.fmin(function, bestSolution, ftol=minError, maxfun=2*dim, full_output = True, retall = True)
    
    for p in pop:
        # insert last DS population into pop dynaimc
        lastP = popDynamic[-1]
        lastP[bestIndex] = p
        popDynamic.append(lastP)
        # insert last DS values into fe dynamic
        lastFE = FEDynamic[-1]
        lastFE[bestIndex] = function(p)
        FEDynamic.append(lastFE)
        # append FDynamic
        lastF = FDynamic[-1]
        FDynamic.append(lastF)
        # append CRDynamic
        lastCR = CRDynamic[-1]
        CRDynamic.append(lastCR)
        
    return (popDynamic, FEDynamic, FDynamic, CRDynamic)


if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    population = 100*np.random.rand(4,2)
    minError = -20
    maxFeval = 10**2
    H = 100
    p = 0.3
    c = 0.5
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = memeticpJADEadaptive(population, \
    tf.sphere, minError, maxFeval)
    plt.semilogy(FEDynamic)
    