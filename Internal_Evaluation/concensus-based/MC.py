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
for mat_file in mat_file_list:
# for mat_file in arff_file_list:
    print(mat_file)
    #22222222222222222222222222222222#
    # mat =  mat_file + '.csv'
    mat =  mat_file + '.mat.csv'
    mat_X = mat_file + '.mat_X.csv'
    mat_y = mat_file + '.mat_y.csv'

    #333333333333333333333333333333333333333#
    t0 = time()
    # df = pd.read_csv(os.path.join("scores_arff", mat), names=headers)    
    df = pd.read_csv(os.path.join("scores_mat", mat), names=headers,low_memory=False)
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
    
    m = len(base_detectors)
    
    similar_mat = np.full((m, m), 1).astype(float)

    for i in keys:
        # get all the configuration with the same hyperparameters
        same_hypers = base_detectors_ranges[i]
        
        for k in same_hypers:
            temp_list = list(range(m))
            temp_list.remove(k)

            for j in temp_list:
                # corr = ndcg_score([np.nan_to_num(output_mat[:, k])], [np.nan_to_num(output_mat[:, j])])
                # corr = spearmanr(output_mat[:, [k, j]])[0]
                corr = kendalltau(output_mat[:, k], output_mat[:, j])[0]
                similar_mat[k, j] = corr
    
    B = (similar_mat+similar_mat.T)/2
    # fix nan problem
    B = np.nan_to_num(B)
    
    similarity = (np.sum(B, axis=1)-1)/(m-1)
    t1 = time()
    duration = round(t1 - t0, ndigits=4)
    
    time_tracker.append(duration)
    print(time_tracker)
    print('kendall')
    # #4444444444444444444444444444444444444444444444#
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.NDCG1.csv'), similarity, delimiter=',')  

    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC1.kendall.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'.MC.kendall.csv'), similarity, delimiter=',') 
    
    # np.savetxt(os.path.join('scores_mat', mat_file+'.MC.csv'), similarity, delimiter=',')
    # np.savetxt(os.path.join('scores_arff', mat_file+'MC.csv'), similarity, delimiter=',')    
    # # y=pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR1.csv')).to_numpy()
    # # w = pd.read_csv(os.path.join("scores_arff", 'Annthyroid.UDR.csv')).to_numpy()
    # # p = np.concatenate([w,y], axis=1)
    # # spearmanr(p)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        