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
import testFunctions as tf

def downhillsimplex(population, function, minError, maxFeval): 
    '''
    implementation of a memetic JADE: \n
    maxFeval-2*dim of the function evaluation are spend on JADE
    2*dim of the function evaluation is used to perform a downhill simplex
    internal parameters of JADE are set to p=0.3 and c=0.5
    
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
    startSolution = population[np.random.randint(0, high=psize)]
    
    _, _, _, _, _, allvecs = sc.optimize.fmin(function, startSolution, ftol=minError, \
                                              maxfun=maxFeval, \
                                              full_output = True, retall = True)
    
    FDynamic = []
    CRDynamic = []
    popDynamic = []
    FEDynamic = []
    for x in allvecs:
        popDynamic.append(np.array([x]))
        
    FEDynamic.append(function(allvecs[-1]))
        
    return (popDynamic, FEDynamic, FDynamic, CRDynamic)


if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt

    population = 100*np.random.rand(4,2)
    minError = 10**-200
    maxFeval = 10**3
    H = 100
    p = 0.3
    c = 0.5
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = downhillsimplex(population, \
    tf.sphere, minError, maxFeval)
    plt.semilogy(FEDynamic)
    