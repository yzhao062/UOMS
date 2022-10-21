# -*- coding: utf-8 -*-

import os
import numpy as np
from scipy.stats import spearmanr,kendalltau
from sklearn.metrics import ndcg_score



# mat_file = 'annthyroid'


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

for mat_file in mat_file_list:

    sp = np.loadtxt(mat_file+'.MC.csv')
    nd = np.loadtxt(mat_file+'.MC.ndcg.csv')
    kd = np.loadtxt(mat_file+'.MC.kendall.csv')
    
    # sp = np.loadtxt(mat_file+'.UDR.csv')
    # nd = np.loadtxt(mat_file+'.UDR.ndcg.csv')
    # kd = np.loadtxt(mat_file+'.UDR.kendall.csv')
    
    # sp = np.loadtxt(mat_file+'.UDR.csv')
    # nd = np.loadtxt(mat_file+'.UDR.ndcg1.csv')
    # kd = np.loadtxt(mat_file+'.UDR.kendall.csv')
    
    
    print(mat_file, ': spearman vs. ndcg', ndcg_score([sp], [nd]))
    print(mat_file, ': spearman vs. kendall', ndcg_score([sp], [kd]))
    print(mat_file, ': ndcg vs. kendall', ndcg_score([nd], [kd]))
    
    print()