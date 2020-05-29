# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:10:24 2020

@author: Nicolai
----------------
"""

import numpy as np
import time
from multiprocessing import Pool
import pJADEFunctions as pF
import testFunctions as tf

def pJADE(population, p, c, function, minError, maxFeval):
    '''
    implementation of pJADE based on: \n
    JADE: Adaptive Differential Evolution with Optional External Archiv \n
    by Zhang and Sanderson
    runs the function evaluation of the population parallel
    
    Parameters
    ----------
    population: numpy array
        2D numpy array where lines are candidates and colums is the dimension
    p: float 
        ]0,1] percentage of best individuals for current-to-p-best mutation
    c: float 
        [0, 1]value for calculating the new muF and muCE
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
    >>> maxError = -1*np.inf
    >>> maxGen = 10**3
    >>> c = 0.5
    >>> population = 100*np.random.rand(50,2)
    >>> p = 0.1
    >>> (popDynamic, FEDynamic, FDynamic, CRDynamic) = 
        JADE(population, p, c, sphere, maxError, maxGen)
    
    '''
    # initialisation of variables 
    populationSize, dimension = population.shape
    functionValue = np.asarray([function(candidate) for candidate in population])
    feCount = populationSize
    F = 0.5
    CR = 0.5
    archive = np.array([population[0]])
    # expection value for parameter self adaption
    muCR = 0.5
    muF = 0.5
    
    popDynamic = []
    FEDynamic = []
    FDynamic = []
    CRDynamic = []
    
    popDynamic.append(np.copy(population))
    FEDynamic.append(np.copy(functionValue))
    FDynamic.append(F)
    CRDynamic.append(CR)
    
    # set up parallel processing pool
    pool = Pool(processes= populationSize if populationSize <= 5 else 5)

    while(feCount < maxFeval and np.min(functionValue) > minError):
        # success history S for control parameters 
        sCR = []
        sF = []
        
        parallelResults = []
        for i in range(populationSize):
            parallelResults.append(pool.apply_async(pF.parallelPopulation, \
                                   args=(muF, muCR, population, archive, \
                                         i, functionValue, p, function)))
        
        for i in range(populationSize):
            (ind, fun) = parallelResults[i].get()
            if(fun < functionValue[i]):
                # build and remove archive
                archLength, _ = archive.shape
                if (archLength >= populationSize):
                    randIndex = np.random.randint(0, high=archLength)
                    archive = np.delete(archive, randIndex, 0)
                archive = np.vstack([archive, population[i]])
                
                # create parameter success history and 
                sF.append(F)
                sCR.append(CR)
                
                # perform selection
                population[i] = ind
                functionValue[i] = fun
        
        # update mean for control parameters
        muCR = (1-c) * muCR + c * arithmeticMean(sCR, muCR)
        muF = (1-c) * muF + c * lehmerMean(sF, muF)
            
        feCount = feCount + populationSize
        popDynamic.append(np.copy(population))
        CRDynamic.append(CR)
        FDynamic.append(F)
        FEDynamic.append(np.copy(functionValue))
        
    pool.close()
    pool.join()
        
    return (popDynamic, FEDynamic, FDynamic, CRDynamic) 
    
    
def lehmerMean(history, muF):
    K = len(history)
    if K != 0:
        sumNumerator = 0
        sumDenominator = 0
        for k in range(K):
            sumNumerator += history[k]*history[k]
            sumDenominator += history[k]
        return sumNumerator/sumDenominator
    else: 
        return muF


def arithmeticMean(history, muCR):
    if len(history) != 0: 
        return np.mean(history)
    else: 
        return muCR
    


if __name__ == "__main__": 

    population = 100*np.random.rand(40,20)
    p = 0.3
    maxError = 10**-200
    maxFE = 5*10**2
    c = 0.5
    
    import JADE
    t1 = time.time()
    JADE.JADE(population, p, c, tf.sphere, maxError, maxFE)
    t1_end = time.time()
    
    t2 = time.time()
    pJADE(population, p, c, tf.sphere, maxError, maxFE)
    t2_end = time.time()
    
    print("time to run JADE: " + str(t1_end - t1))
    print("time to run pJADE: " + str(t2_end - t2))
    
    
    
    
    
    