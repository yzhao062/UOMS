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
from sklearn.utils import check_array
import arff

# Define data file and read X and y
mat_file_list = [
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

def read_arff(file_path, misplaced_list):
    misplaced = False
    for item in misplaced_list:
        if item in file_path:
            misplaced = True

    file = arff.load(open(file_path))
    data_value = np.asarray(file['data'])
    attributes = file['attributes']

    X = data_value[:, 0:-2]
    if not misplaced:
        y = data_value[:, -1]
    else:
        y = data_value[:, -2]
    y[y == 'no'] = 0
    y[y == 'yes'] = 1
    y = y.astype('float').astype('int').ravel()

    if y.sum() > len(y):
        print(attributes)
        raise ValueError('wrong sum')

    return X, y, attributes

misplaced_list = ['Arrhythmia', 'Cardiotocography', 'Hepatitis', 'ALOI',
                  'KDDCup99']
arff_list = [
    os.path.join('semantic', 'Annthyroid', 'Annthyroid_withoutdupl_07.arff'),
    os.path.join('semantic', 'Arrhythmia', 'Arrhythmia_withoutdupl_46.arff'),
    os.path.join('semantic', 'Cardiotocography',
                  'Cardiotocography_withoutdupl_22.arff'),
    os.path.join('semantic', 'HeartDisease',
                  'HeartDisease_withoutdupl_44.arff'),
    # os.path.join('semantic', 'Hepatitis', 'Hepatitis_withoutdupl_16.arff'),
    os.path.join('semantic', 'InternetAds',
                  'InternetAds_withoutdupl_norm_19.arff'),
    os.path.join('semantic', 'PageBlocks', 'PageBlocks_withoutdupl_09.arff'),
    os.path.join('semantic', 'Pima', 'Pima_withoutdupl_35.arff'),
    os.path.join('semantic', 'SpamBase', 'SpamBase_withoutdupl_40.arff'),
    os.path.join('semantic', 'Stamps', 'Stamps_withoutdupl_09.arff'),
    os.path.join('semantic', 'Wilt', 'Wilt_withoutdupl_05.arff'),

    os.path.join('literature', 'ALOI', 'ALOI_withoutdupl.arff'),
    os.path.join('literature', 'Glass', 'Glass_withoutdupl_norm.arff'),
    os.path.join('literature', 'PenDigits',
                  'PenDigits_withoutdupl_norm_v01.arff'),
    os.path.join('literature', 'Shuttle', 'Shuttle_withoutdupl_v01.arff'),
    os.path.join('literature', 'Waveform', 'Waveform_withoutdupl_v01.arff'),
    os.path.join('literature', 'WBC', 'WBC_withoutdupl_v01.arff'),
    os.path.join('literature', 'WDBC', 'WDBC_withoutdupl_v01.arff'),
    os.path.join('literature', 'WPBC', 'WPBC_withoutdupl_norm.arff')
]

moving_size = 3

perf_mat = np.zeros([len(mat_file_list), 3])

time_tracker = []
for j in range(len(mat_file_list)):
    mat_file = mat_file_list[j]
    mat_file_path = os.path.join("data", "DAMI", arff_list[j])

    X, y, attributes = read_arff(mat_file_path, misplaced_list)
    
    X = X.astype('float64')
    y = y.ravel()
    score_mat = np.loadtxt(os.path.join("scores_arff", mat_file+'.csv'), delimiter=',')
    
    t0 = time()
    rank_mat = rankdata(score_mat, axis=0)
    inv_rank_mat = 1 / rank_mat

    
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
    last_kendall = kendall_tracker[-1]
    
    # # option 2: moving avg
    # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
    
    # # option 3: average of all
    # last_kendall = np.mean(kendall_tracker)
    
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
            last_kendall = kendall_tracker[-1]
            
            # # option 2: moving avg
            # last_kendall = np.mean(kendall_tracker[-1*moving_size:])
            
            # # option 3: average of all
            # last_kendall = np.mean(kendall_tracker)

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
    # np.savetxt(os.path.join('scores_arff', mat_file+'.SELECT.ALL.Model.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.SELECT.ALL.Target.csv'), final_target*-1, delimiter=',')