# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
from time import time
import itertools

# temporary solution for relative imports in case pyod is not installed
# if pyod is installed, no need to use the following line
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname("__file__"), '..')))
# supress warnings for clean output
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import loadmat

from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.cof import COF
from pyod.models.sod import SOD

from pyod.utils.utility import standardizer
from pyod.utils.utility import precision_n_scores
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import arff
from sklearn.utils import check_array

from base_detectors import get_detectors
from em import em, mv, get_em_mv, get_em_mv_original

# TODO: add neural networks, LOCI, SOS, COF, SOD

# Define data file and read X and y
mat_file_list = [
    'Annthyroid',
    'Arrhythmia',
    'Cardiotocography',
    'HeartDisease',  # too small
    'Hepatitis',  # too small
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


full_param = []
full_param_headers = []

base_detectors, randomness_flags =  get_detectors()
param_tracker = 0

classifier_name = "LODA"
param_list = []
param_list_1 = [5, 10, 15, 20, 25, 30]
param_list_2 = [10, 20, 30, 40, 50, 75, 100, 150, 200]

for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))

classifier_name = "ABOD"
param_list = [3, 5, 10, 15, 20, 25, 50]
for r in param_list:
    param_tracker+=1
    
full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "IForest"
param_list = []
param_list_1 = [10, 20, 30, 40, 50, 75, 100, 150, 200]
param_list_2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "kNN"
param_list = []
param_list_1 = ['largest', 'mean', 'median']
param_list_2 = [1, 5 ,10, 15, 20, 25, 50, 60, 70, 80, 90, 100]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "LOF"
param_list = []
param_list_1 = ['manhattan', 'euclidean', 'minkowski']
param_list_2 = [1, 5 ,10, 15, 20, 25, 50, 60, 70, 80, 90, 100]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "HBOS"
param_list = []
param_list_1 = [5, 10, 20, 30, 40, 50, 75, 100]
param_list_2 = [0.1, 0.2, 0.3, 0.4, 0.5]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))


classifier_name = "OCSVM"
param_list = []
param_list_1 = ["linear", "poly", "rbf", "sigmoid"]
param_list_2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for r in itertools.product(param_list_1, param_list_2): 
    param_list.append((r[0], r[1]))
    param_tracker+=1

full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))

classifier_name = "COF"
param_list = [3, 5, 10, 15, 20, 25, 50]
for r in param_list:
    param_tracker+=1
full_param.extend(param_list)
full_param_headers.extend([classifier_name]*len(param_list))

combined_list = [(str(a), b) for a,b in zip(full_param_headers, full_param)]

assert (len(combined_list)==len(base_detectors))


#%%
n_classifiers = len(base_detectors)

df_columns = ['Data', '#Samples', '# Dimensions', 'Outlier Perc']
df_columns.extend(combined_list)

# initialize the container for saving the results
roc_df = pd.DataFrame(columns=df_columns)
prn_df = pd.DataFrame(columns=df_columns)
ap_df = pd.DataFrame(columns=df_columns)
time_df = pd.DataFrame(columns=df_columns)
time2_df = pd.DataFrame(columns=df_columns)

score_files = []


for j in range(len(mat_file_list)):

    mat_file = mat_file_list[j]
    mat_file_path = os.path.join("data", "DAMI", arff_list[j])

    X, y, attributes = read_arff(mat_file_path, misplaced_list)
    X = check_array(X).astype('float64')
    y = y.ravel()

    outliers_fraction = np.count_nonzero(y) / len(y)
    outliers_percentage = round(outliers_fraction * 100, ndigits=4)

    # construct containers for saving results
    roc_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    prn_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    ap_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    time_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    time2_list = [mat_file, X.shape[0], X.shape[1], outliers_percentage]
    
    score_mat = np.zeros([X.shape[0], n_classifiers])
    em_mat = np.zeros([n_classifiers, ])
    mv_mat = np.zeros([n_classifiers, ])

    
    for i, clf in enumerate(base_detectors):
        # define the number of iterations
        n_ite = 1
        
        if randomness_flags[i]:
            n_ite = 5
        
        roc_intermediate = []
        prn_intermediate = []
        ap_intermediate = []
        time_intermediate = []
        time_intermediate2 = []
        
        score_intermediate = np.zeros([X.shape[0], n_ite])
        
        em_intermediate = np.zeros([n_ite, ])
        mv_intermediate = np.zeros([n_ite, ])
        
        for j in range(n_ite):
            print("\n... Processing", mat_file, '...', combined_list[i], 'Iteration', j + 1)
            random_state = np.random.RandomState(j)
            X_norm = standardizer(X)
            
            t0 = time()
            clf.fit(X_norm)
            test_scores = clf.decision_scores_
            # test_scores = np.nan_to_num(test_scores, nan=np.mean(test_scores))
            test_scores[np.isnan(test_scores)] = np.nanmean(test_scores)

            t1 = time()
            duration = round(t1 - t0, ndigits=4)
    
    
            # roc and other metrics should not be averaged
            # roc_intermediate.append(round(roc_auc_score(y, test_scores), ndigits=4))
            # prn_intermediate.append(round(precision_n_scores(y, test_scores), ndigits=4))
            # ap_intermediate.append(round(average_precision_score(y, test_scores), ndigits=4))
            time_intermediate.append(duration)
            
            score_intermediate[:, j] = test_scores
            em_intermediate[j], mv_intermediate[j] = get_em_mv_original(X_norm, clf)
            t2 = time()
            duration = round(t2 - t1, ndigits=4)
            time_intermediate2.append(duration)
        
        # roc_list.append(np.mean(roc_intermediate))
        # prn_list.append(np.mean(prn_intermediate))
        # ap_list.append(np.mean(ap_intermediate))
        # time_list.append(np.mean(time_intermediate))
    
        # use the averaged score to calculate other metrics
        score_mat[:, i] = np.mean(score_intermediate, axis=1)
        em_mat[i] = np.mean(em_intermediate)
        mv_mat[i] = np.mean(mv_intermediate)
        
        roc_list.append(round(roc_auc_score(y, score_mat[:, i]), ndigits=4))
        prn_list.append(round(precision_n_scores(y, score_mat[:, i]), ndigits=4))
        ap_list.append(round(average_precision_score(y, score_mat[:, i]), ndigits=4))
        time_list.append(np.mean(time_intermediate))
        time2_list.append(np.mean(time_intermediate2))
    # index = np.argmin(np.abs(np.array(w)-np.mean(w)))
    
    # add a save to local
    score_files.append(score_mat)
    np.savetxt(os.path.join("scores_arff", mat_file+'2.csv'), score_mat, delimiter=',')
    np.savetxt(os.path.join("scores_arff", mat_file+'.em.csv'), em_mat, delimiter=',')
    np.savetxt(os.path.join("scores_arff", mat_file+'.mv.csv'), mv_mat, delimiter=',')

    # time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time_list).transpose()
    temp_df.columns = df_columns
    time_df = pd.concat([time_df, temp_df], axis=0)
    
    # time_list = time_list + np.mean(time_mat, axis=0).tolist()
    temp_df = pd.DataFrame(time2_list).transpose()
    temp_df.columns = df_columns
    time2_df = pd.concat([time2_df, temp_df], axis=0)

    # roc_list = roc_list + np.mean(roc_mat, axis=0).tolist()
    temp_df = pd.DataFrame(roc_list).transpose()
    temp_df.columns = df_columns
    roc_df = pd.concat([roc_df, temp_df], axis=0)

    # prn_list = prn_list + np.mean(prn_mat, axis=0).tolist()
    temp_df = pd.DataFrame(prn_list).transpose()
    temp_df.columns = df_columns
    prn_df = pd.concat([prn_df, temp_df], axis=0)

    # ap_list = ap_list + np.mean(ap_mat, axis=0).tolist()
    temp_df = pd.DataFrame(ap_list).transpose()
    temp_df.columns = df_columns
    ap_df = pd.concat([ap_df, temp_df], axis=0)
    
    # if file does not exist write header
    time_df.to_csv('time_arff_2.csv', index=False, float_format='%.3f')
    time2_df.to_csv('time2_arff_2.csv', index=False, float_format='%.3f')
    roc_df.to_csv('roc_arff_2.csv', index=False, float_format='%.3f')
    prn_df.to_csv('prns_arff_2.csv', index=False, float_format='%.3f')
    ap_df.to_csv('ap_arff_2.csv', index=False, float_format='%.3f')
