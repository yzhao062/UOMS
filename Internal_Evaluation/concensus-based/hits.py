# -*- coding: utf-8 -*-

import numpy as np
import sys

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import average_precision_score, roc_auc_score
from time import time
# from sklearn.datasets import one_class_data

from scipy.io import loadmat
from scipy.stats import rankdata
import os
from scipy.stats import kendalltau

mat_file_list = [
    'annthyroid.mat',
    'arrhythmia.mat',
    'breastw.mat',
    'glass.mat',
    'ionosphere.mat',
    'letter.mat',
    'lympho.mat',
    'mammography.mat',
    'mnist.mat',
    'musk.mat',
    'optdigits.mat',
    'pendigits.mat',
    'pima.mat',
    'satellite.mat',
    'satimage-2.mat',
    # 'shuttle.mat',
    # 'smtp_n.mat',
    'speech.mat',
    'thyroid.mat',
    'vertebral.mat',
    'vowels.mat',
    'wbc.mat',
    'wine.mat',
]

arff_file_list = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    # 'Hepatitis',  # too small
    'InternetAds',
    'PageBlocks',
    'Pima',
    'SpamBase',
    'Stamps',
    'Wilt',
    'ALOI', # too large
    'Glass', # too small
    'PenDigits',
    'Shuttle',
    'Waveform',
    'WBC', # too small
    'WDBC', # too small
    'WPBC', # too small
]

# moving_size = 3
perf_mat = np.zeros([len(mat_file_list), 3])

time_tracker = []
for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    # loading and vectorization
    mat = loadmat(os.path.join("data", "ODDS", mat_file))
    score_mat = np.loadtxt(os.path.join("scores_mat", mat_file+'.csv'), delimiter=',')

    t0 = time()
    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat

    X = mat['X']
    y = mat['y'].ravel()
    
    n_samples, n_models = score_mat.shape[0], score_mat.shape[1]

    hub_vec = np.full([n_models, 1],  1/n_models)
    auth_vec = np.zeros([n_samples, 1])
    
    hub_vec_list = []
    auth_vec_list = []
    
    hub_vec_list.append(hub_vec)
    auth_vec_list.append(auth_vec)
    
    for i in range(500):
        auth_vec = np.dot(inv_rank_mat, hub_vec)
        auth_vec = auth_vec/np.linalg.norm(auth_vec)
        
        # update hub_vec
        hub_vec = np.dot(inv_rank_mat.T, auth_vec)
        hub_vec = hub_vec/np.linalg.norm(hub_vec)
        
        # stopping criteria
        auth_diff = auth_vec - auth_vec_list[-1]
        hub_diff = hub_vec - hub_vec_list[-1]
        
        # print(auth_diff.sum(), auth_diff.mean(), auth_diff.std())
        # print(hub_diff.sum(), hub_diff.mean(), hub_diff.std())
        # print()
        
        if np.abs(auth_diff.sum()) <= 1e-10 and np.abs(auth_diff.mean()) <= 1e-10 and np.abs(hub_diff.sum()) <= 1e-10 and np.abs(hub_diff.mean()) <= 1e-10:
            print('break at', i)
            break
        
        auth_vec_list.append(auth_vec)
        hub_vec_list.append(hub_vec)
        
    
    print('HITS', mat_file, roc_auc_score(y, auth_vec*-1), 
          average_precision_score(y, auth_vec*-1), 
          precision_n_scores(y, auth_vec*-1))
    print()
    
    perf_mat[j, 0] = roc_auc_score(y, auth_vec*-1)
    perf_mat[j, 1] = average_precision_score(y, auth_vec*-1)
    perf_mat[j, 2] = precision_n_scores(y, auth_vec*-1)

    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Model.csv'), hub_vec, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.HITS.Target.csv'), auth_vec, delimiter=',')