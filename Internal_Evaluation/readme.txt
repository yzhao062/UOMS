Supplementary Mateiral for submission "A Large-scale Study on Unsupervised Outlier Model Selection".


Installation and Dependency:

Simply run "pip install -r requirements.txt"


data: contains the 39 datasets: DAMI (18 datasets) ans ODDS (21 datasets)


stand-alone:
- IREOS
- MASS-VOLUME (MV) AND EXCESS-MASS (EM)
- Clustering Validation Metrics

consensus-based:
- UDR
- MC
- MCS
- MC by HITS
- Unsupervisd OD Ensembling

analysis: contains the intermediate result and corresponding analysis



To reproduce the paper:
1. Copy the scripts (score_generator_arff.py and score_generator_mat.py) to the root dir, and then execute them. The generated score files will be saved in scores_mat and scores_arff folders. You could skip this step as we have uploaded all the score files (/analysis/OD_Intrinsic_Measure). Note: some OD algorithms, e.g., iForest and LODA, come with randomness, so the results might be slightly different if you rerun it.
2. To get the baseline performance, e.g., UDR's results, please first get the corresponding baseline files from stand-alone or consensus-based folder, and execute them from the root directory. The score files will be saved in scores_mat and scores_arff folders. You could skip this step as we have uploaded all the results.
3. To conduct statistical analysis, you could use the results and the scripts in analysis folder.

