#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:22:38 2021

@author: celinejin
"""

# from scipy.stats import norm
from scipy import stats
from scipy.stats import multivariate_normal
import numpy as np

def surprise_threshold(X, gpr):
    '''
    Computes the SS surprise threshold at points X[-1] based on existing samples X
    and Y using a Gaussian process surrogate model.
    
    Args:
        X[-1]: Points at which EI shall be computed (m x d).
        X: Sample locations (n x d).
        Y: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        
    Returns:
        SS threshold at points X[-1].
    '''
    mu, sigma = gpr.predict(X[-1].reshape(-1,1), return_std=True)
    Y_cont1 = (mu+1.5*sigma).flat[0]
    mu=list(mu.flat)
    sigma=list(sigma.flat)
    sh=multivariate_normal.pdf(Y_cont1, mean=mu, cov=sigma)
    sh=np.append(sh,1)    
    surprise_threshold_value=sum(-np.log(sh))
    return surprise_threshold_value


# def surprise_threshold_ndim(X_next, gpr):
#     mu, sigma = gpr.predict(X_next, return_std=True)
#     Y_cont1 = (mu+2*sigma).flat[0]
#     #mu=list(mu.flat)
#     #sigma=list(sigma.flat)
#     #sh=multivariate_normal.pdf(Y_cont1, mean=mu, cov=sigma)
#     #sh=np.append(sh,1)    
#     #surprise_threshold_value=sum(-np.log(sh))
#     m_dist_x = (Y_cont1-mu)/sigma
#     m_dist_x = np.dot(m_dist_x, (Y_cont1-mu))
#     surprise_threshold_value=-np.log(1-stats.chi2.cdf(m_dist_x, 2))
#     return surprise_threshold_value