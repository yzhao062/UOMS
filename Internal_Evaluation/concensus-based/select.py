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

moving_size = 3
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
    
    # build target vector 
    target = np.mean(inv_rank_mat, axis=1)
    
    kendall_vec = np.full([n_models,], -99).astype(float)
    kendall_tracker = []
    
    model_ind = list(range(n_models))
    selected_ind = []
    last_kendall = 0
    
    # build the first target
    for i in model_ind:
        kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
    
    most_sim_model = np.argmax(kendall_vec)
    kendall_tracker.append(np.max(kendall_vec))
    
    # option 1: last one: keep increasing/non-decreasing
    # last_kendall = kendall_tracker[-1]
    
    # # option 2: moving avg
    # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
    
    # option 3: average of all
    last_kendall = np.mean(kendall_tracker)
    
    selected_ind.append(most_sim_model)
    model_ind.remove(most_sim_model)
    
    
    while len(model_ind) != 0:
    
        target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
        kendall_vec = np.full([n_models,], -99).astype(float)
        
        for i in model_ind:
            kendall_vec[i] = kendalltau(target, inv_rank_mat[:, i])[0]
            
        most_sim_model = np.argmax(kendall_vec)
        max_kendall = np.max(kendall_vec)
        
        if max_kendall >= last_kendall:
            selected_ind.append(most_sim_model)
            model_ind.remove(most_sim_model)
            kendall_tracker.append(max_kendall)
            
            # option 1: last one: keep increasing/non-decreasing
            # last_kendall = kendall_tracker[-1]
            
            # # option 2: moving avg
            # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
            
            # option 3: average of all
            last_kendall = np.mean(kendall_tracker)

        else:
            break
    
    final_target = np.mean(inv_rank_mat[:, selected_ind], axis=1)
    average_target = np.mean(inv_rank_mat, axis=1)
    
    
    print('SELECT', mat_file, roc_auc_score(y, final_target*-1), 
          average_precision_score(y, final_target*-1), 
          precision_n_scores(y, final_target*-1))
    print('Average', mat_file, roc_auc_score(y, average_target*-1), 
          average_precision_score(y, average_target*-1), 
          precision_n_scores(y, average_target*-1))
    print()
    
    perf_mat[j, 0] = roc_auc_score(y, final_target*-1)
    perf_mat[j, 1] = average_precision_score(y, final_target*-1)
    perf_mat[j, 2] = precision_n_scores(y, final_target*-1)
    
    similarity = []
    for k in range(n_models):
        similarity.append(kendalltau(final_target, inv_rank_mat[:, k])[0])
        
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Model.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_mat', mat_file+'.SELECT.ALL.Target.csv'), final_target*-1, delimiter=',')