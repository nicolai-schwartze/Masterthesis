# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:10:24 2020

@author: Nicolai
----------------
"""

import numpy as np
import time
from scipy.stats import cauchy
import testFunctions as tf

def JADE(population, p, c, function, minError, maxFeval):
    '''
    implementation of JADE based on: \n
    JADE: Adaptive Differential Evolution with Optional External Archiv \n
    by Zhang and Sanderson
    
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
    # temorary arrays for holding the population and its function values
    # during a generation
    trailPopulation = np.copy(population)
    trailFunctionValue = np.copy(functionValue)
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

    while(feCount < maxFeval and np.min(functionValue) > minError):
        # success history S for control parameters 
        sCR = []
        sF = []
        
        for i in range(populationSize):
            F = selectF(muF)
            vi = mutationCurrentToPBest1(population, archive, i, functionValue, F, p)
            
            CR = selectCR(muCR)
            ui = crossoverBIN(np.array([population[i]]), vi, CR)
            
            trailPopulation[i] = ui
            ####################################################
            # for actual JADE missing constraint handling here #
            ####################################################
            trailFunctionValue[i] = function(ui)
        
            if(trailFunctionValue[i] < functionValue[i]):
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
                population[i] = trailPopulation[i]
                functionValue[i] = trailFunctionValue[i]
        
        # update mean for control parameters
        muCR = (1-c) * muCR + c * arithmeticMean(sCR, muCR)
        muF = (1-c) * muF + c * lehmerMean(sF, muF)
            
        feCount = feCount + populationSize
        popDynamic.append(np.copy(population))
        CRDynamic.append(CR)
        FDynamic.append(F)
        FEDynamic.append(np.copy(functionValue))
        
    return (popDynamic, FEDynamic, FDynamic, CRDynamic)


def crossoverBIN(xi, vi, CR):
    r, c = vi.shape
    K = np.random.randint(low=0, high=c)
    ui = []
    for j in range(c):
        if j==K or np.random.rand() < CR:
            ui.append(vi[0][j])
        else:
            ui.append(xi[0][j])
    return np.asarray(ui)


def mutationCurrentToPBest1(population, archive, currentIndex, functionValue, F, p): 
    popSize, dim = population.shape
    bestPIndizes = functionValue.argsort()[0:int(p*popSize)]
    bestIndex = np.random.choice(bestPIndizes)
    currentCandidate = np.array([population[currentIndex]])
    bestCandidate = np.array([population[bestIndex]])
    if(bestIndex == currentIndex):
        population = np.delete(population, bestIndex, 0)
    elif(bestIndex < currentIndex):
        population = np.delete(population, bestIndex, 0)
        population = np.delete(population, currentIndex-1, 0)
    elif(bestIndex > currentIndex):
        population = np.delete(population, bestIndex, 0)
        population = np.delete(population, currentIndex, 0)
    popUnion = unionRowVec(population, archive)
    maxIndex, _ = popUnion.shape
    indizes = [i for i in range(maxIndex)]
    if not(len(indizes) == 1):
        indizes = np.random.permutation(indizes)
        r0 = np.array([popUnion[indizes[0]]])
        r1 = np.array([popUnion[indizes[1]]])
        vi = currentCandidate + F*(bestCandidate - currentCandidate) + F*(r0 - r1)
    else: 
        vi = currentCandidate + F*(bestCandidate - currentCandidate)
    return vi


def unionRowVec(A, B):
    nrows, ncols = A.shape
    # construct a data type so that the array is a "one-dimensional" array of lists
    dtype = (', '.join([str(A.dtype)]*ncols))
    # construct a "view" (or in other words "convert") the array to the new dtype
    # use the "one-dimensional" array to calculate the one-dimensional union
    C = np.union1d(A.view(dtype), B.view(dtype))
    # reconstruct a "two-dimensional" array 
    C = C.view(A.dtype).reshape(-1, ncols)
    return C


def selectCR(muCR):
    # get new normaly distributed CR
    newCR = np.random.normal(loc=muCR, scale=0.1)
    # if the value for the new CR is outside of [0,1]
    # replace with the closest boundary
    if newCR < 0: 
        return 0
    elif newCR > 1: 
        return 1
    else:
        return newCR
    
    
def selectF(muF):
    # randomly generate new F based on a Cauchy distribution
    newF = cauchy.rvs(loc=muF, scale=0.1)
    # if it is smaller than 0, try to generate a new F value that is not 
    # smaller than 0
    if newF <= 0: 
        while newF <= 0:
            newF = cauchy.rvs(loc=muF, scale=0.1)
    # when the newly calculated F value is greater than 1 -> return 1
    if newF > 1: 
        return 1
    else: 
        return newF
    
    
    
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
    np.seterr("raise")
    print("start test")
    
    A = np.random.randint(0, high=5, size=(10, 2))
    B = np.random.randint(0, high=5, size=(10, 2))
    
    t1 = time.time()
    C = unionRowVec(A, B)
    print("union of two matrices: " + str(time.time() - t1)) 
    
    population = 100*np.random.rand(4,2)
    archive = 500*np.random.rand(4,2)
    functionValue = np.asarray([tf.sphere(candidate) for candidate in population])
    F = 0.5
    p = 0.3
    t1 = time.time()
    mutationCurrentToPBest1(population, archive, 0, functionValue, F, p)
    print("time to execute mutation: " + str(time.time() - t1))
    
    maxError = 10**-80
    maxGen = 10**3
    c = 0.5
    t1 = time.time()
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = JADE(population, p, c, tf.sphere, maxError, maxGen)
    print("time to run JADE: " + str(time.time() - t1))
    plt.plot(FDynamic)
    
    
    
    
    
    