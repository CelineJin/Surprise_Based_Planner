#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:42:56 2021

@author: celinejin
"""

from scipy.stats import norm
from scipy import stats
from scipy.stats import multivariate_normal
import numpy as np

# def surprise(X_next, gpr):
def surprise(X, Y, gpr):
    '''
    Computes the SS suprise at current point X[-1] based on existing samples X
    and Y using a Gaussian process surrogate model.
    
    Args:
        X[-1]: Points at which SS surprise shall be computed (m x d).
        X: Sample locations (n x d).
        Y: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        
    
    Returns:
        SS surprise at points X.
    '''
    mu, sigma = gpr.predict(X[-1].reshape(-1,1), return_std=True)
    mu=list(mu.flat)
    sigma=list(sigma.flat)
    # Y_cont = f(X_next)
    # Y_cont,ind, X_cont = f(data,X_next)
    Y_cont = Y[-1]
    sh=multivariate_normal.pdf(Y_cont, mean=mu, cov=sigma)
    sh=np.append(sh,1)    
    surprise_value=sum(-np.log(sh))
    return surprise_value

# def surprise_ndim(data, X_next, gpr):
#     '''
#     Computes the EI at points X based on existing samples X_sample
#     and Y_sample using a Gaussian process surrogate model.
    
#     Args:
#         X: Points at which EI shall be computed (m x d).
#         X_sample: Sample locations (n x d).
#         Y_sample: Sample values (n x 1).
#         gpr: A GaussianProcessRegressor fitted to samples.
#         xi: Exploitation-exploration trade-off parameter.
    
#     Returns:
#         Expected improvements at points X.
#     '''
#     mu, sigma = gpr.predict(X_next, return_std=True)
#     mu=list(mu.flat)
#     sigma=list(sigma.flat)
#     # Y_cont = f(X_next)
#     Y_cont,ind, X_cont = fn(data,X_next)
#     sh=multivariate_normal.pdf(Y_cont, mean=mu, cov=sigma)
#     sh=np.append(sh,1)    
#     surprise_value=sum(-np.log(sh))
#     return surprise_value


def kl_divergence(p, q):
    # return np.sum(np.where(q != 0, p * np.log(p / q), 0))
    return np.sum(np.where(q!=0, np.log(p/q),0))

def bayesian_surprise(X, Y, n, m):
    Y_cont = Y[-1]
    
    mu, sigma = n.predict(X[-1].reshape(1,-1), return_std=True)
    mu=list(mu.flat)
    sigma=list(sigma.flat)
    sh=multivariate_normal.pdf(Y_cont, mean=mu, cov=sigma)
    
    mu1, sigma1 = m.predict(X[-1].reshape(1,-1), return_std=True)
    mu1=list(mu1.flat)
    sigma1=list(sigma1.flat)
    sh1=multivariate_normal.pdf(Y_cont, mean=mu1, cov=sigma1)
    return abs(kl_divergence(sh1, sh))

# def bayesian_surprise_ndim(data, X_next, n, m):
#     Y_cont, ind_cont, X_cont = fn(data, X_next)
#     mu, sigma = n.predict(X_next, return_std=True)
#     m_dist_x = (Y_cont-mu)/sigma
#     m_dist_x = np.dot(m_dist_x, (Y_cont-mu))
#     su1=-np.log(1-stats.chi2.cdf(m_dist_x, 2))
    
#     Y_cont1, ind_cont1, X_cont1 = fn(data, X_next)
#     mu1, sigma1 = m.predict(X_next, return_std=True)
#     m1_dist_x = (Y_cont1-mu1)/sigma1
#     m1_dist_x = np.dot(m1_dist_x, (Y_cont1-mu1))
#     su2=-np.log(1-stats.chi2.cdf(m1_dist_x, 2))
#     return abs(kl_divergence(su1, su2))
