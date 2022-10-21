#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from s_dbw import S_Dbw
import sys


# In[2]:


def hubert(x, c):
    n = len(x)
    x = np.array(x)
    c = np.array(c)
    
    diff = abs(x - x[:,None])
    diff_c = abs(c - c[:,None])
    
    mask = np.zeros_like(diff, dtype=np.bool)
    mask[np.tril_indices_from(mask)] = True

    return ((mask * diff) * (diff_c) * 2).sum() / (n * (n-1))

def i_index(x, c, c_normal, c_anomaly):
    max_d = abs(c_normal - c_anomaly)
    n = len(x)
    x = np.array(x)
    c = np.array(c)
    numerator = (abs(x-x.mean()).sum())
    denominator = (abs(x-c).sum())
    return ((numerator * max_d) / (denominator * 2)) ** 2

def Dunn(x, c_binary):
    x = np.array(x)
    c_binary = np.array(c_binary)
    normal_mask = np.tile(c_binary, (len(c_binary), 1))
    anomaly_mask = np.tile(1 - c_binary.reshape(-1, 1).T, (len(c_binary), 1))
    distance = abs(x - x[:,None])
    
    diff_inter = distance * normal_mask * (anomaly_mask.T) # distance of samples from different clusters
    diff_intra_1 = distance * normal_mask * (normal_mask.T) # distance of samples in cluster 1
    diff_intra_2 = distance * anomaly_mask * (anomaly_mask.T) # distance of samples in cluster 2

    min_value = np.min(diff_inter[np.nonzero(diff_inter)])
    max_value = max(np.max(diff_intra_1), np.max(diff_intra_2))
    return min_value/max_value
    
def xb(x, c, c_normal, c_anomaly):
    return sum((x-c)**2) / (len(x) * ((c_normal - c_anomaly) ** 2))


# In[3]:


def get_statistics(x, anomaly_ratio, num_attribute):
    n = len(x)
    p = num_attribute
    x = x.to_numpy()
    num_anomaly = int(np.around(anomaly_ratio * len(x) / 100))
    x_ranked = np.sort(x)
    threshold = x_ranked[-num_anomaly]
    c_normal = np.mean(x_ranked[0:-num_anomaly])
    c_anomaly = np.mean(x_ranked[-num_anomaly:])
    # c is the cluster center for each data point (two cluster centers in this case, and one sample corresponds to one of them.)
    c = [c_anomaly if i >= threshold else c_normal for i in x]
    c_binary = [1 if i >= threshold else 0 for i in x]
    r2 = metrics.r2_score(y_true=x, y_pred=c) # Should be correct
    std = np.sqrt(metrics.mean_squared_error(y_true=x, y_pred=c) * (n / ((n-2) * p))) # Should be correct; rescale according to the paper
    h = hubert(x, c) # Should be correct
    ch = metrics.calinski_harabasz_score(X=x.reshape(len(x), 1), labels=c_binary) # Should be correct
    s = metrics.silhouette_score(X=x.reshape(len(x), 1), labels=c_binary) # Should be correct
    i = i_index(x, c, c_normal, c_anomaly) # should be correct
    db = metrics.davies_bouldin_score(X=x.reshape(len(x), 1), labels=c_binary) # Should be correct
    xbs = xb(x, c, c_normal, c_anomaly) # should be correct
    sd = S_Dbw(X=x.reshape(len(x), 1), labels=c_binary) # need to double check
    dunn = Dunn(x, c_binary) # Should be correct
    # return r2, std, h, ch, s, i, db, xbs, sd
    return r2, std, h, ch, s, i, db, xbs, sd, dunn


# In[6]:


for c in ['wine',
'lympho',
'glass',
'vertebral',
'ionosphere',
'wbc',
'arrhythmia',
'breastw',
'pima',
'vowels',
'letter',
'musk',
'speech',
'thyroid',
'optdigits',
'satimage-2',
'satellite',
'pendigits',
'annthyroid',
'mnist',
'mammography']:    
    dataset = c
    dataset_dir = "OD_Intrinsic_Measure/scores_mat/{}".format(dataset)
    score = pd.read_csv('{}.mat.csv'.format(dataset_dir), header=None)
    ap_csv = pd.read_csv('OD_Intrinsic_Measure/ap_mat.csv')
    ap_csv = ap_csv.set_index('Data')
    score.columns = ap_csv.columns[3:]
    
    #average precision
    #roc_auc_score
    #precision_n_scores

    # rank of ap
    ap_csv_rank = ap_csv.iloc[:,3:].copy()
    for i in range(len(ap_csv_rank.index)):
        ap_csv_rank.iloc[i] = ap_csv_rank.iloc[i].rank(pct=True)

    # rank of roc
    roc_csv = pd.read_csv('OD_Intrinsic_Measure/roc_mat.csv')
    roc_csv = roc_csv.set_index('Data')
    roc_csv_rank = roc_csv.iloc[:,3:].copy()
    for i in range(len(roc_csv_rank.index)):
        roc_csv_rank.iloc[i] = roc_csv_rank.iloc[i].rank(pct=True)

    # rank of prn
    prn_csv = pd.read_csv('OD_Intrinsic_Measure/prns_mat.csv')
    prn_csv = prn_csv.set_index('Data')
    prn_csv_rank = prn_csv.iloc[:,3:].copy()
    
    for i in range(len(prn_csv_rank.index)):
        prn_csv_rank.iloc[i] = prn_csv_rank.iloc[i].rank(pct=True)
    #######################################
    ap = []

    r2 = []
    std = []
    h = []
    ch = []
    s = []
    i_indices = []
    db = []
    xbs = []
    sd = []
    dunn = []
    for i in range(len(score.columns)):
        #print(i)
        statistics = get_statistics(score.iloc[:,i], ap_csv.loc[dataset, 'Outlier Perc'], ap_csv.loc[dataset, '# Dimensions'])
        r2.append(statistics[0])
        std.append(statistics[1])
        h.append(statistics[2])
        ch.append(statistics[3])
        s.append(statistics[4])
        i_indices.append(statistics[5])
        db.append(statistics[6])
        xbs.append(statistics[7])
        sd.append(statistics[8])
        dunn.append(statistics[9])
        ap.append(ap_csv_rank.iloc[:,i][dataset])
    
    # save
    df = pd.DataFrame(columns = ap_csv_rank.columns)
    df['Metric'] = ['r2', 'std', 'h', 'ch', 's', 'i', 'db', 'xbs', 'sd', 'dunn']
    df = df.set_index('Metric')
    df.loc['r2'] = r2
    df.loc['std'] = std
    df.loc['h'] = h
    df.loc['ch'] = ch
    df.loc['s'] = s
    df.loc['i'] = i_indices
    df.loc['db'] = db
    df.loc['xbs'] = xbs
    df.loc['sd'] = sd
    df.loc['dunn'] = dunn
    df.to_csv('{}_metrics.csv'.format(dataset))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




