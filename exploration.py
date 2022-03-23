#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:23:26 2021

@author: celinejin
"""

import numpy as np
# def exploration(b, stepsize):

#     X_CHECK = list()
#     X_DIST = list()
#     b.sort(axis=0)
#     for i_x in range (b.shape[0]-1):
#         for x_sel in np.arange(float(b[i_x]), float(b[i_x+1]-stepsize), stepsize):
#             if x_sel>=float(b[i_x]) and x_sel<float(b[i_x])+(float(b[i_x+1])-float(b[i_x]))/2:
#                 x_dist = x_sel - float(b[i_x]);
#             else:
#                 x_dist = float(b[i_x+1]) - x_sel;
        
#             X_CHECK.append(x_sel) ;
#             X_DIST.append(x_dist);

#     X_expl=np.array([X_CHECK[np.argmax(X_DIST)]])
#     return X_expl.reshape((-1,1))

# def furthest_node(node, nodes):
#     dist_2 = np.sum((nodes - node)**2, axis=1)
#     return np.argmax(dist_2)

# def furthest_exploration(X_all, X_sampled, X_next):
#     a1_rows = set(map(tuple, X_all))
#     a2_rows = set(map(tuple, X_sampled))
#     X_current=np.array(list(a1_rows.difference(a2_rows)))
#     if len(X_current)>1:
#         X_next_sug= X_current[furthest_node(X_next,X_current), :].reshape(1,-1)
#     elif len(X_current)==0:
#         X_next_sug=X_all[np.random.choice(X_all.shape[0], size=1, replace=False), :]   
#     else:
#         X_next_sug=X_current
#     return X_next_sug

def max_var(nodes, gpr):
    mu, sigma = gpr.predict(nodes, return_std=True)
    return np.min(np.argmax(sigma))

def maxvar_exploration(X_all, X_sampled, gpr, note):
    a1_rows = set(map(tuple, X_all))
    a2_rows = set(map(tuple, X_sampled))
    X_current=np.array(list(a1_rows.difference(a2_rows)))
    if len(X_current) > 1:
        X_next_sug = X_current[max_var(X_current, gpr), :].reshape(1,-1)
    elif len(X_current) ==0:
        note.append('There is no point left in the LHD space.')
        X_next_sug = X_all[np.random.choice(X_all.shape[0],size=1,replace=False),:]
    else:
        note.append('There is no point left in the LHD space.')
        X_next_sug = X_current
    return X_next_sug, note


