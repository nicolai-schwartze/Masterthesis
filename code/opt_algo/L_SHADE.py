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

def L_SHADE(population, p, H, function, minError, maxGeneration):
    ''' 
    implementation of L-SHADE based on: \n
    Improving the Search Performance of SHADE Using Linear Population Size Reduction\n 
    by Tanabe and Fukunaga\n
    adaptions: 
        * no constraint handling implemented 
        * population size reduction based on generation insteas of function evaluation 
    
    
    Parameters
    ----------
    population: numpy array
        2D numpy array where lines are candidates and colums is the dimension
        
    p: float ]0,1]
        percentage of best individuals for current-to-p-best mutation
    H: int 
        size of the memory
    function: function
        fitness function that is optimised
    minError: float 
        stopping condition on function value
    maxGeneration: int 
        stopping condition on max number of generation
    
    Returns
    -------
    history: tuple
        tupel[0] - popDynamic\n
        tupel[1] - FEDynamic\n
        tupel[2] - FDynamic\n
        tupel[3] - CRDynamic\n
    
    Examples
    --------
    >>> import numpy as np
    >>> def sphere(x):
            return np.dot(x,x)
    >>> maxError = -1*np.inf
    >>> maxGen = 10**3
    >>> H = 50
    >>> population = 100*np.random.rand(50,2)
    >>> p = 0.1
    >>> (popDynamic, FEDynamic, FDynamic, CRDynamic) = 
        L_SHADE(population, p, H, sphere, maxError, maxGen)
    
    
    
    '''
    # initialisation of variables 
    populationSize, dimension = population.shape
    functionValue = np.asarray([function(candidate) for candidate in population])
    genCount = 1
    F = 0.5
    CR = 0.5
    archive = np.array([population[0]])
    # temorary arrays for holding the population and its function values
    # during a generation
    trailPopulation = np.copy(population)
    trailFunctionValue = np.copy(functionValue)
    # memory for control parameters
    mCR = 0.5*np.ones(H)
    mF  = 0.5*np.ones(H)
    # k is the running memory index
    k = 0
    # population size reduction parameter
    NGmin = int(np.ceil(1/p))
    NGinit = populationSize
    
    popDynamic = []
    FEDynamic = []
    FDynamic = []
    CRDynamic = []
    
    popDynamic.append(np.copy(population))
    FEDynamic.append(np.copy(functionValue))
    FDynamic.append(np.copy(mF))
    CRDynamic.append(np.copy(mCR))

    while(genCount < maxGeneration and np.min(functionValue) > minError):
        # success history S for control parameters 
        sCR = []
        sF = []
        sCRtemp = []
        sFtemp = []
        
        for i in range(populationSize):
            F = selectF(mF)
            sFtemp.append(F)
            vi = mutationCurrentToPBest1(population, archive, i, functionValue, F, p)
            
            CR = selectCR(mCR)
            sCRtemp.append(CR)
            ui = crossoverBIN(np.array([population[i]]), vi, CR)
            
            trailPopulation[i] = ui
            #######################################################
            # for actual L-SHADE missing constraint handling here #
            #######################################################
            trailFunctionValue[i] = function(ui)
        
        functionValueDifference = []
        for i in range(populationSize):
            if(trailFunctionValue[i] <= functionValue[i]):
                # build and remove archive
                archLength, _ = archive.shape
                if (archLength >= populationSize):
                    randIndex = np.random.randint(0, high=archLength)
                    archive = np.delete(archive, randIndex, 0)
                archive = np.vstack([archive, population[i]])
                
                # create parameter success history and weights for lehmer mean
                sF.append(sFtemp[i])
                sCR.append(sCRtemp[i])
                # equation 9 in paper
                functionValueDifference.append(np.abs(trailFunctionValue[i] - functionValue[i]))
                
                # perform selection
                population[i] = trailPopulation[i]
                functionValue[i] = trailFunctionValue[i]
        
        # calculate lehmer weights
        weights = []
        sDF = np.sum(functionValueDifference)
        for df in functionValueDifference:
            if sDF == 0.0:
                weights.append(0)
            else: 
                weights.append(df/sDF)
        
        # update parameter memory with success history
        if len(sCR) != 0 and len(sF) != 0:
            if mCR[k] == np.inf or np.max(mCR) == 0:
                mCR[k] = np.inf
            else: 
                mCR[k] = weightedLehmermean(sCR, weights)
            mF[k] = weightedLehmermean(sF, weights)
            k += 1
            if k >= H: k = 0
            
        # perform population size reduction
        # calculate new population size based on the current generation count
        NG_1 = populationSizeReduction(genCount, maxGeneration, NGinit, NGmin)
        # if the new population should be smaller
        if NG_1 < populationSize:
            # delete worst individuals from the population
            functionValueSorted = np.argsort(functionValue)
            indizesToRemove = functionValueSorted[-int(populationSize-NG_1):]
            population = np.delete(population, indizesToRemove, 0)
            functionValue = np.delete(functionValue, indizesToRemove)
            populationSize = population.shape[0]
            # resize archive to the population size by deleting random indizes
            while archive.shape[0] > populationSize:
                randi = np.random.randint(0, high=archive.shape[0])
                archive = np.delete(archive, randi, 0)


        genCount = genCount + 1
        print("generation: {}".format(genCount))
        popDynamic.append(np.copy(population))
        CRDynamic.append(np.copy(mCR))
        FDynamic.append(np.copy(mF))
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


def selectCR(mCR):
    # select random index from the CR memory
    ri = np.random.randint(0, mCR.shape[0])
    # if the selected CR is the terminal character -> return 0
    if mCR[ri] == np.inf:
        return 0
    # otherwise calculate a random CR based on a normal distribution
    # wih mCR[ri] as the centre
    else: 
        newCR = np.random.normal(loc=mCR[ri], scale=0.1)
    # if the value for the new CR is outside of [0,1]
    # replace with the closest boundary
    if newCR < 0: 
        return 0
    elif newCR > 1: 
        return 1
    else:
        return newCR
    
    
def selectF(mF):
    # get a random index ri in the F memory
    ri = np.random.randint(0, mF.shape[0])
    # get random new F by a chauchy distribution where mF[ri] is the centre
    newF = cauchy.rvs(loc=mF[ri], scale=0.1)
    # if it is smaller than 0, try to generate a new F value that is not 
    # smaller than 0
    if newF <= 0: 
        while newF <= 0:
            newF = cauchy.rvs(loc=mF[ri], scale=0.1)
    # when the newly calculated F value is greater than 1 -> return 1
    if newF > 1: 
        return 1
    else:
        return newF
    
    
def weightedLehmermean(history, weights):
    K = len(history)
    sumNumerator = 0
    sumDenominator = 0
    for k in range(K):
        sumNumerator += weights[k]*history[k]*history[k]
        sumDenominator += weights[k]*history[k]
    
    if sumDenominator == 0.0: 
        return 0.0
    else: 
        return sumNumerator/sumDenominator
    
    
    
def populationSizeReduction(genCounter, maxGen, NGinit, NGmin): 
    NG_1 = np.round(((NGmin - NGinit)/maxGen)*genCounter + NGinit)
    return NG_1



if __name__ == "__main__": 
    np.seterr("raise")
    print("start test")
    
    A = np.random.randint(0, high=5, size=(10, 2))
    B = np.random.randint(0, high=5, size=(10, 2))
    
    t1 = time.time()
    C = unionRowVec(A, B)
    print("union of two matrices: " + str(time.time() - t1)) 
    
    population = 100*np.random.rand(50,2)
    archive = 500*np.random.rand(50,2)
    functionValue = np.asarray([tf.sphere(candidate) for candidate in population])
    F = 0.5
    p = 0.1
    t1 = time.time()
    mutationCurrentToPBest1(population, archive, 0, functionValue, F, p)
    print("time to execute mutation: " + str(time.time() - t1))
    
    maxError = -1*np.inf
    maxGen = 10**3
    H = 50
    t1 = time.time()
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = L_SHADE(population, p, H, tf.sphere, maxError, maxGen)
    print("time to run L-SHADE: " + str(time.time() - t1))
    print("optimum: " + str(np.min(FEDynamic[-1])))
    
    
    