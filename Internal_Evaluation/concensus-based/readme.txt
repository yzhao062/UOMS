This repository includes the implementation of corrsponding concensus-based algorithms.

Specifically, 

UDR: UDR.py
MCS: UDR.py and set p=\sqrt(279)~= 18
MC: MC.py
HITS: hits.py and hits_arff.py
Unsupervised Outlier Model Ensembling: select.py and select_arff.py

It is noted that for UDR and MC, we may use kendall tau, spearman rho, or NDCG as the similarity measure.
Consequently, multiple versions are included in the implementation. Please comment out the unnecessary part while suited.

Running these scripts at the root level will output the results to scores_mat and scores_arff folder.