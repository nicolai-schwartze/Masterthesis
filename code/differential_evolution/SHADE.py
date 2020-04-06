# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 15:10:24 2020

@author: Nicolai
----------------
"""

import numpy as np
import time
from scipy.stats import cauchy

def SHADE(population, p, H, function, minError, maxGeneration):
    '''
    input: 
    population    - 2D numpy array where lines are candidates and colums is the dimension
    p             - percentage of best individuals for current-to-p-best mutation
    H             - size of the memory
    minError      - stopping condition on function value
    maxGeneration - stopping condition on max number of generation
    
    output: 
    tupel[1] - popDynamic
    tupel[2] - FEDynamic
    tupel[3] - FDynamic
    tupel[4] - CRDynamic
    
    based on: Success-History Based Parameter Adaptation for Differential Evolution 
    by Tanabe and Fukunaga
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
            #####################################################
            # for actual SHADE missing constraint handling here #
            #####################################################
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
            
        genCount = genCount + 1
        print(genCount)
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


if __name__ == "__main__": 
    np.seterr("raise")
    print("start test")
    
    A = np.random.randint(0, high=5, size=(10, 2))
    B = np.random.randint(0, high=5, size=(10, 2))
    
    t1 = time.time()
    C = unionRowVec(A, B)
    print("union of two matrices: " + str(time.time() - t1)) 
    
    def sphere(x):
        return np.dot(x,x)
    
    population = 100*np.random.rand(50,2)
    archive = 500*np.random.rand(50,2)
    functionValue = np.asarray([sphere(candidate) for candidate in population])
    F = 0.5
    p = 0.3
    t1 = time.time()
    mutationCurrentToPBest1(population, archive, 0, functionValue, F, p)
    print("time to execute mutation: " + str(time.time() - t1))
    
    maxError = 10**-80
    maxGen = 10**3
    H = 3
    A = 20
    t1 = time.time()
    (popDynamic, FEDynamic, FDynamic, CRDynamic) = SHADE(population, p, H, sphere, maxError, maxGen)
    print("time to run SHADE: " + str(time.time() - t1))
    print("optimum: " + str(np.min(FEDynamic[-1])))
    
    
    
    
    
    