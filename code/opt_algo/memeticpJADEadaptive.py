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

def memeticpJADEadaptive(startPopulation, function, minError, maxFeval): 
    '''
    implementation of an adaptive kernel memetic pJADEadaptive: \n
    maxFeval-2*dim of the function evaluation are started on pJADEadaptive
    2*dim of the function evaluation is used to perform a downhill simplex
    when the pJADEadaptive terminates, the donwhill simplex is started 
    if the previous best result is outperformed, the dim and pop size is increased
    if the new result is not better than the best result, a restart is perfomed
    internal parameters of pJADEadaptive are set to p=0.3 and c=0.5, delaytime = 100
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
    psize, dim = startPopulation.shape
    p = 0.3
    c = 0.5
    delayTime = 100
    kernelsize = 0
    if dim % 4 == 0:
        kernelsize = 4
    elif dim % 6 == 0:
        kernelsize = 6
    else:
        raise ValueError("dimensionality error: must be multiple of kerenlsize 4 or 6")
        
    populationFactor = int(psize/dim)
    
    fecounter = 0
    globalPopDynamic = []
    globalFEDynamic = []
    globalFDynamic = []
    globalCRDynamic = []
    
    bestFE = np.inf
    bestPop = None
    
    
    while fecounter < maxFeval:
        print("starting with dimension: " + str(dim))
        jadePop, jadeFe, jadeF, jadeCr = pJADEadaptive(startPopulation, p, c, delayTime, function, minError, maxFeval-fecounter-(2*dim))
        fecounter += (len(jadePop) * psize)
        print("finished JADE with " + str((fecounter/maxFeval) * 100) + "% FE")
        print("="*45)
        print("start direct search with downhill simplex")
        bestIndex = np.argmin(jadeFe[-1])
        bestSolution = jadePop[-1][bestIndex]
        _, _, _, _, _, dsPop  = sc.optimize.fmin(function, bestSolution, ftol=minError, maxfun=2*dim, full_output = True, retall = True)
        fecounter += 2*dim
        print("finished DS with " + str((fecounter/maxFeval) * 100) + "% FE")
        for pop in dsPop:
            # insert last DS population into pop dynaimc
            lastP = jadePop[-1]
            lastP[bestIndex] = pop
            jadePop.append(lastP)
            # insert last DS values into fe dynamic
            lastFE = jadeFe[-1]
            lastFE[bestIndex] = function(pop)
            jadeFe.append(lastFE)
            # append FDynamic
            lastF = jadeF[-1]
            jadeF.append(lastF)
            # append CRDynamic
            lastCR = jadeCr[-1]
            jadeCr.append(lastCR)
    
        # concatinate to global return arguments
        globalPopDynamic.extend(jadePop.copy())
        globalFEDynamic.extend(jadeFe.copy())
        globalFDynamic.extend(jadeF.copy())
        globalCRDynamic.extend(jadeCr.copy())

        # if FE value is better than before
        if np.min(globalFEDynamic[-1]) < np.min(bestFE):
            bestFE = globalFEDynamic[-1]
            bestPop = np.copy(globalPopDynamic[-1])
            # increase dimensionality
            print(" --> increase dimensionality")
            print("="*45)
            startPopulation = np.hstack((globalPopDynamic[-1], np.random.randn(psize, kernelsize)))
            psize, dim = startPopulation.shape
            additionalInd = np.random.randn((populationFactor * dim) - psize, dim)
            startPopulation = np.vstack((startPopulation, additionalInd))
            psize, dim = startPopulation.shape
            
        else:
            print(" --> reduce and restart")
            print("="*45)
            # reduce population size
            newdim = dim-kernelsize
            if newdim <= bestPop.shape[1]:
                startPopulation = np.copy(bestPop) + np.random.randn(bestPop.shape[0], bestPop.shape[1])
                # update dim and psize
                psize, dim = startPopulation.shape
            else:
                startPopulation = np.copy(globalPopDynamic[-1][0:2*(dim-kernelsize),0:newdim])
                # update dim and psize
                psize, dim = startPopulation.shape
                # add random normal distributed variation to account for restart
                startPopulation += np.random.randn(psize,dim)
            
    
    # if the bestPop and bestFE where ever changed, append to the return value
    # needed to make sol_kernel correct
    if (not (type(bestPop) == type(None))):
        globalPopDynamic.append(bestPop)
        globalFEDynamic.append(bestFE)
        globalFDynamic.append(globalFDynamic[-1])
        globalCRDynamic.append(globalCRDynamic[-1])

    return (globalPopDynamic, globalFEDynamic, globalFDynamic, globalCRDynamic)


if __name__ == "__main__": 
    
    import matplotlib.pyplot as plt
    
    population = 100*np.random.rand(8,4)
    minError = 0
    maxFeval = 10**3
    p = 0.3
    c = 0.5
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = memeticpJADEadaptive(population, \
    tf.sphere, minError, maxFeval)
    print(FEDynamic[-1])
    
    goaldim = 0
    for i in FEDynamic:
        if i.shape[0] > goaldim:
            goaldim = i.shape[0]
    plotList = []
    for i in FEDynamic:
        currentdim = i.shape[0]
        plotList.append(np.lib.pad(i, (0,goaldim-currentdim), 'constant', constant_values=(0)))
        
    plt.semilogy(plotList)
    