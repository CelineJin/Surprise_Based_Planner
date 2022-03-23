#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 10:03:18 2021

@author: celinejin
"""

# import json
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import numpy as np
# import pickle
# from scipy.spatial.distance import cdist
from pyDOE import lhs

# Self defined
from exploration import maxvar_exploration
from exploitation import closest_node0
from surprises import surprise, bayesian_surprise
from surprise_threshold import surprise_threshold


class cam1planner():
    
    def planner(rawdata):

        # json_file = open('2019-11-25-135707.json','r+')
        # rawdata = json.load(json_file)
        # paramnames = rawdata['ParamNames']
        # X_current = np.array(data['prime-delay'])
        # Y_current = np.array(data['objective'])
        # surprise_method = data['method']
        
        surprise_method = rawdata['metric'][0]
        cam1par = np.vstack(rawdata['History'])
        xdim = cam1par.shape[1]-1
        X_all = lhs(xdim, samples=200, criterion = 'maximin')
        X_all = X_all * rawdata['MaxVals'][0]
        
        # X_current = newinput[0]
        # Y_current = newinput[1]
        # newinput = list([X_current.flat[0], Y_current.flat[0]])
        # rawdata["History"].append(newinput)
        # json_file.seek(0)
        # json.dump(rawdata, json_file)
        # json_file.close()
        
        cam1par_past = cam1par[0:-1]
        
        X_past = np.vstack(cam1par_past[:,0])
        Y_past = np.vstack(cam1par_past[:,1])
        
        # X = np.append(X_past, np.array([[X_current]]), axis=0) 
        # Y = np.append(Y_past, np.array([[Y_current]]), axis=0) 
        
        X = np.vstack(cam1par[:,0])
        Y = np.vstack(cam1par[:,1])
        
        
        ## Gaussian process with Mat??rn kernel as surrogate model
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        gpr = GaussianProcessRegressor(kernel=m52, alpha=0.10)
        gpr_past = GaussianProcessRegressor(kernel=m52, alpha=0.10)
        
        
        
        g_SS = list()
        s_SS = list()
        g_BS = list()
        
        # z = np.copy(X)
        
        ## If it is initial run, run exploration to generate next experiment
        if len(Y) <= 3:
            X_next_SS = maxvar_exploration(X_all, X, gpr)
            X_next_BS = maxvar_exploration(X_all, X, gpr)
            X_next = X_next_SS
            note = ['Initializing']
        else: ## explore or exploit by surprise
            gpr.fit(X,Y) ## for SS AND BS surprise
            gpr_past.fit(X_past, Y_past)  ## for BS surprise
            
            if surprise_method == 'Shannon':
                note = ['Shannon']
                ## Shannon surprise-based acquisition
                a = surprise(X, Y, gpr_past)
                p = surprise_threshold(X,gpr_past)
                g_SS.append(a)
                s_SS.append(p)
                if a>p: ## surprise happens, go for exploitation: find neighbor
                    note.append('exploitation')
                    # mindist = cdist(X[-1],X[:-1],metric='euclidean').min()
                    # mindist = abs(X[-1]-X[:-1]).min()
                    # if  mindist > 0.5:
                    #     X_next_SS = np.array([[list(X[-1].flat)[0]-stepsize]])
                    # else:
                    #     X_next_SS = np.array([[list(X[-1].flat)[0]-(mindist/2)]])
                    X_next_SS, note = closest_node0(X[-1], X[:-1], rawdata['MaxVals'][0],note)
                else: ## go for exploration: max variance of lhs candidates
                    note.append('exploration')
                    X_next_SS, note = maxvar_exploration(X_all, X, gpr, note)
                if X_next_SS > rawdata['MaxVals'][0]:
                    note.append('Error: recommended X is beyond the bounds of X. Please stop printing!')
                    X_next = rawdata['MaxVals'][0]
                else:
                    X_next = X_next_SS 
             
            if surprise_method == 'Bayesian':   
                note = ['Bayesian']
                ## Bayesian surprise-based acquisition
                a = bayesian_surprise(X, Y, gpr_past, gpr)
                g_BS.append(a)
                if abs(a) < 0.005: ## surprise happens, go for exploitation: find neighbor
                    note.append('exploitation')
                    # mindist = cdist(X[-1],X[:-1],metric='euclidean').min()
                    # mindist = abs(X[-1]-X[:-1]).min()
                    # if mindist > 0.5:
                    #     X_next_BS = np.array([[list(X[-1].flat)[0]-stepsize]])
                    # else:
                    #     X_next_BS = np.array([[list(X[-1].flat)[0]-(mindist/2)]])
                    X_next_BS, note = closest_node0(X[-1], X[:-1], rawdata['MaxVals'][0],note)
                else: ## go for exploration: max variance of lhs candidates
                    note.append('exploration')
                    X_next_BS, note = maxvar_exploration(X_all, X, gpr, note)
                if X_next_BS > rawdata['MaxVals'][0]:
                    note.append('Error: recommended X is beyond the bounds of X. Please stop printing!')
                    X_next = rawdata['MaxVals'][0]
                else:
                    X_next = X_next_BS 
                    
        X_next = X_next.reshape(1,-1)
                
        predicted_objvalue = gpr.predict(X_next, return_std=False, return_cov=False)
            
        return X_next, predicted_objvalue, note    
    

# specificmodel = cam1planner()
# pickle.dump(specificmodel, open('model.pkl','wb'))
