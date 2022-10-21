# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import ndcg_score



# mat_file = 'annthyroid'

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


for mat_file in arff_file_list:

    # sp = np.loadtxt(mat_file+'.MC.csv')
    # nd = np.loadtxt(mat_file+'.MC.ndcg.csv')
    # kd = np.loadtxt(mat_file+'.MC.kendall.csv')
    
    # sp = np.loadtxt(mat_file+'.UDR.csv')
    # nd = np.loadtxt(mat_file+'.UDR.ndcg.csv')
    # kd = np.loadtxt(mat_file+'.UDR.kendall.csv')
    
    sp = np.loadtxt(mat_file+'.UDR.csv')
    nd = np.loadtxt(mat_file+'.UDR.ndcg1.csv')
    kd = np.loadtxt(mat_file+'.UDR.kendall.csv')
    
    
    print(mat_file, ': spearman vs. ndcg', ndcg_score([sp], [nd]))
    print(mat_file, ': spearman vs. kendall', ndcg_score([sp], [kd]))
    print(mat_file, ': ndcg vs. kendall', ndcg_score([nd], [kd]))
    
    print()