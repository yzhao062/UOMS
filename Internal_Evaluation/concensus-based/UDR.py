# -*- coding: utf-8 -*-

import os 
from time import time

import pandas as pd
import numpy as np
from base_detectors import get_detectors
from copy import deepcopy
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import rankdata
clf_df = pd.read_csv('roc_mat_2.csv', low_memory=False)
headers = clf_df.columns.tolist()[4:]


mat_file_list = [
    'annthyroid',
    'arrhythmia',
    'breastw',
    'glass',
    'ionosphere',
    'letter',
    'lympho',
    'mammography',
    'mnist',
    'musk',
    'optdigits',
    'pendigits',
    'pima',
    'satellite',
    'satimage-2',
    'speech',
    'thyroid',
    'vertebral',
    'vowels',
    'wbc',
    'wine',
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
time_tracker = []
#1111111111111111111111111111111#
# for mat_file in mat_file_list:
for mat_file in arff_file_list:
    print(mat_file)
    #22222222222222222222222222222222#
    mat =  mat_file + '.csv'
    # mat =  mat_file + '.mat.csv'
    mat_X = mat_file + '.mat_X.csv'
    mat_y = mat_file + '.mat_y.csv'

    t0 = time()
    #333333333333333333333333333333333333333#
    df = pd.read_csv(os.path.join("scores_arff", mat), names=headers)    
    # df = pd.read_csv(os.path.join("scores_mat", mat), names=headers, low_memory=False)
    output_mat = df.to_numpy().astype('float64')
    output_mat = np.nan_to_num(output_mat)
    
    output_mat_r = rankdata(output_mat, axis=0)
    output_mat = MinMaxScaler().fit_transform(output_mat_r)
    base_detectors, randomness_flags =  get_detectors()
    
    base_detectors_ranges = {}
    

    keys = list(range(8))
    base_detectors_ranges[0] = list(range(0, 54))
    base_detectors_ranges[1] = list(range(54, 61))
    base_detectors_ranges[2] = list(range(61, 142))
    base_detectors_ranges[3] = list(range(142, 178))
    base_detectors_ranges[4] = list(range(178, 214))
    base_detectors_ranges[5] = list(range(214, 254))
    base_detectors_ranges[6] = list(range(254, 290))
    base_detectors_ranges[7] = list(range(290, 297))
    
    sum_check = 0
    key_len = []
    
    for i in keys:
        sum_check += len(base_detectors_ranges[i])
        key_len.append(len(base_detectors_ranges[i]))
    assert (sum_check==297)
    
    p = np.min(key_len)-1
    # p = 18
    m = len(base_detectors)
    
    similar_mat = np.zeros((m, p))


###############################################################################
# ### Kendall version 0    
#     for i in keys:
#         # get all the configuration with the same hyperparameters
#         same_hypers = base_detectors_ranges[i]
        
#         for k in same_hypers:
#             temp_list = deepcopy(same_hypers) 
#             temp_list.remove(k)
#             # print(temp_list)
#             p_list = np.random.choice(temp_list, size=p, replace=False)
#             # print(p_list)
#             # print()
            
#             for j, value in enumerate(p_list):
#                 # print(j, value)
#                 corr1 = kendalltau(output_mat[:, k], output_mat[:, value])[0]
#                 corr2 = kendalltau(output_mat[:, value], output_mat[:, k])[0]
#                 similar_mat[k, j] = (corr1+corr2)/2
###############################################################################


# ###############################################################################
# # ### Kendall version 1    

#     for i in keys:
#         # get all the configuration with the same hyperparameters
#         same_hypers = base_detectors_ranges[i]
        
#         for k in same_hypers:
#             temp_list = list(range(m))
#             temp_list.remove(k)
#             # print(temp_list)
#             p_list = np.random.choice(temp_list, size=p, replace=False)
            
#             for j, value in enumerate(p_list):
#                 # print(j, value)
#                 corr1 = kendalltau(output_mat[:, k], output_mat[:, value])[0]
#                 corr2 = kendalltau(output_mat[:, value], output_mat[:, k])[0]
#                 similar_mat[k, j] = (corr1+corr2)/2
# # ###############################################################################

# ##############################################################################
# # ### NDCG version 0    
#     for i in keys:
#         # get all the configuration with the same hyperparameters
#         same_hypers = base_detectors_ranges[i]
        
#         for k in same_hypers:
#             temp_list = deepcopy(same_hypers) 
#             temp_list.remove(k)
#             # print(temp_list)
#             p_list = np.random.choice(temp_list, size=p, replace=False)
#             # print(p_list)
#             # print()
            
#             for j, value in enumerate(p_list):
#                 # print(k, value)
#                 # print(j, value)
#                 # corr1 = ndcg_score([np.nan_to_num(output_mat[:, k])], [np.nan_to_num(output_mat[:, value])])
#                 # corr2 = ndcg_score([np.nan_to_num(output_mat[:, value])], [np.nan_to_num(output_mat[:, k])])
#                 # similar_mat[k, j] = (corr1+corr2)/2
#                 corr1 = ndcg_score(np.nan_to_num(output_mat[:, k]).reshape(1,-1), np.nan_to_num(output_mat[:, value]).reshape(1,-1))
#                 corr2 = ndcg_score(np.nan_to_num(output_mat[:, value]).reshape(1,-1), np.nan_to_num(output_mat[:, k].reshape(1,-1)))
#                 similar_mat[k, j] = (corr1+corr2)/2
##############################################################################


###############################################################################
# ### MCS NDCG
#     for i in keys:
#         # get all the configuration with the same hyperparameters
#         same_hypers = base_detectors_ranges[i]
        
#         for k in same_hypers:
#             temp_list = list(range(m))
#             temp_list.remove(k)
#             # print(temp_list)
#             p_list = np.random.choice(temp_list, size=p, replace=False)
            
#             for j, value in enumerate(p_list):
#                 # print(j, value)
#                 # ndcg_score(, np.nan_to_num(output_mat[:, value].reshape(1,-1)))
#                 corr1 = ndcg_score(np.nan_to_num(output_mat[:, k]).reshape(1,-1), np.nan_to_num(output_mat[:, value]).reshape(1,-1))
#                 corr2 = ndcg_score(np.nan_to_num(output_mat[:, value]).reshape(1,-1), np.nan_to_num(output_mat[:, k].reshape(1,-1)))
#                 similar_mat[k, j] = (corr1+corr2)/2
        
###############################################################################


# ###############################################################################    
    for i in keys:
        # get all the configuration with the same hyperparameters
        same_hypers = base_detectors_ranges[i]
        
        for k in same_hypers:
            temp_list = deepcopy(same_hypers) 
            temp_list.remove(k)
            # print(temp_list)
            p_list = np.random.choice(temp_list, size=p, replace=False)
            # print(p_list)
            # print()
            
            for j, value in enumerate(p_list):
                # print(j, value)
                corr = spearmanr(output_mat[:, [k, value]])
                similar_mat[k, j] = corr[0]
# ##############################################################################

# ###############################################################################
# #######  Variant    
#     for i in keys:
#         # get all the configuration with the same hyperparameters
#         same_hypers = base_detectors_ranges[i]
        
#         for k in same_hypers:
#             temp_list = list(range(m))
#             temp_list.remove(k)
#             # print(temp_list)
#             p_list = np.random.choice(temp_list, size=p, replace=False)
#             # print(p_list)
#             # print()
            
#             for j, value in enumerate(p_list):
#                 # print(j, value)
#                 corr = spearmanr(output_mat[:, [k, value]])
#                 similar_mat[k, j] = corr[0]
# ###############################################################################
    similar_mat = np.nan_to_num(similar_mat)
    similarity = np.median(similar_mat, axis=1)
    
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    time_tracker.append(duration)
    print(time_tracker)
    print('UDR spearman')
    #4444444444444444444444444444444444444444444444#
    # np.savetxt(os.path.join('scores_mat', mat_file+'.UDR.NDCG.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.UDR.NDCG.csv'), similarity, delimiter=',')

    # np.savetxt(os.path.join('scores_mat', mat_file+'.MCS.NDCG.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MCS.NDCG.csv'), similarity, delimiter=',')
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.UDR.kendall.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.UDR.kendall1.csv'), similarity, delimiter=',')  
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MCS.kendall.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MCS.kendall.csv'), similarity, delimiter=',')  

    # np.savetxt(os.path.join('scores_mat', mat_file+'.MCS.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MCS.csv'), similarity, delimiter=',') 
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.UDR1.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.UDR1.csv'), similarity, delimiter=',')    
    # y=pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR1.csv')).to_numpy()
    # w = pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR.csv')).to_numpy()
    # p = np.concatenate([w,y], axis=1)
    # spearmanr(p)
        
        
    
        
        
        
        
        
        
        
        