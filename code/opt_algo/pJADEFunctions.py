# -*- coding: utf-8 -*-
"""
Created on Thu May 28 09:01:13 2020

@author: Nicolai
----------------
"""

from scipy.stats import cauchy
import numpy as np
import time


def parallelPopulation(muF, muCR, population, archive, i, functionValue, p, function):
    
    F = selectF(muF)
    vi = mutationCurrentToPBest1(population, archive, i, functionValue, F, p)
            
    CR = selectCR(muCR)
    ui = crossoverBIN(np.array([population[i]]), vi, CR)
    
    return (ui, function(ui))


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
