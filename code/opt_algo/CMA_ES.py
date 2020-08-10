# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 20:51:52 2020

@author: Nicolai
"""

import numpy as np

def CMA_ES (yp, sigma, sigma_stop, femax, lambda_n, mu, function):
    N = yp.shape[1]
    C = np.eye(N)
    
    sigma_history = [sigma]
    fit_history = [function(yp)]
    fe = 1
    
    while (sigma_history(g) > sigma_stop) and (fe < femax):
        M = np.linalg.cholesky(C)
        for l in range(lambda_n):
            z_l = np.random.randn(1,N)
            d_l = M*z_l
            y_l = y_p * sigma*d_l
            f_l = function(y_l)

