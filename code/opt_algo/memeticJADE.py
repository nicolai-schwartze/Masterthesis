# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 14:03:18 2020

@author: Nicolai
"""

import sys
sys.path.append("../differential_evolution")
from JADE import JADE
import numpy as np
import scipy as sc

def memeticJADE(population, function, minError, maxIter): 
    '''
    implementation of a memetic JADE: \n
    2/3 of the iterations (generations) are spend on JADE
    1/3 of the iterations is used to performa a downhill simplex
    internal parameters of JADE are set to p=0.3 and c=0.5
    
    Parameters
    ----------
    population: numpy array
        2D numpy array where lines are candidates and colums is the dimension
    function: function
        fitness function that is optimised
    minError: float 
        stopping condition on function value
    maxIter: int
        stopping condition on max number of generation
    
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
    popDynamic, FEDynamic, FDynamic, CRDynamic = JADE(population, p, c, function, \
                                                      minError, int(np.ceil((2/3)*maxIter)))
    print("finished JADE")
    print("="*45)
    print("start direct search with downhill simplex")
    bestIndex = np.argmin(FEDynamic[-1])
    bestSolution = popDynamic[-1][bestIndex]
    _, _, _, _, _, pop  = sc.optimize.fmin(function, bestSolution, ftol=minError, \
                                           maxiter=int(np.floor((1/3)*maxIter)), \
                                           full_output = True, retall = True)
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
    
    def sphere(x):
        return np.dot(x,x)
    
    population = 100*np.random.rand(4,2)
    minError = 10**-200
    maxIter = 10**3
    H = 100
    p = 0.3
    c = 0.5
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = memeticJADE(population, \
    sphere, minError, maxIter)
    plt.semilogy(FEDynamic)
    