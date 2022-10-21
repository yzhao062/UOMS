import numpy as np
from sklearn.metrics import auc
from sklearn.utils import shuffle as sh

def em(t, t_max, volume_support, s_unif, s_X, n_generated):
    EM_t = np.zeros(t.shape[0])
    n_samples = s_X.shape[0]
    s_X_unique = np.unique(s_X)
    EM_t[0] = 1.
    for u in s_X_unique:
        # if (s_unif >= u).sum() > n_generated / 1000:
        EM_t = np.maximum(EM_t, 1. / n_samples * (s_X > u).sum() -
                          t * (s_unif > u).sum() / n_generated
                          * volume_support)
    amax = np.argmax(EM_t <= t_max) + 1
    if amax == 1:
        print ('\n failed to achieve t_max \n')
        amax = -1
    AUC = auc(t[:amax], EM_t[:amax])
    return AUC, EM_t, amax


def mv(axis_alpha, volume_support, s_unif, s_X, n_generated):
    n_samples = s_X.shape[0]
    s_X_argsort = s_X.argsort()
    mass = 0
    cpt = 0
    u = s_X[s_X_argsort[-1]]
    mv = np.zeros(axis_alpha.shape[0])
    for i in range(axis_alpha.shape[0]):
        # pdb.set_trace()
        while mass < axis_alpha[i]:
            cpt += 1
            u = s_X[s_X_argsort[-cpt]]
            mass = 1. / n_samples * cpt  # sum(s_X > u)
        mv[i] = float((s_unif >= u).sum()) / n_generated * volume_support
    return auc(axis_alpha, mv), mv


def get_em_mv(X, clf, averaging=5, max_features=5, alpha_min=0.9, 
              alpha_max=0.999, n_generated=100000, t_max = 0.9):
    
    em_clf, mv_clf = 0, 0
    nb_exp = 0
    
    n_features = X.shape[1]
    
    while nb_exp < averaging:
        # print(nb_exp, averaging)
        features = sh(np.arange(n_features))[:max_features]
        X_ = X[:, features]

        lim_inf = X_.min(axis=0)
        lim_sup = X_.max(axis=0)
        volume_support = (lim_sup - lim_inf).prod()
        if volume_support > 0:
            nb_exp += 1
            t = np.arange(0, 100 / volume_support, 0.001 / volume_support)
            axis_alpha = np.arange(alpha_min, alpha_max, 0.001)
            unif = np.random.uniform(lim_inf, lim_sup,
                                     size=(n_generated, max_features))

            clf.fit(X_)
            s_X_clf = clf.decision_scores_ * -1
            s_unif_clf = clf.decision_function(unif) * -1
            
            em_clf += em(t, t_max, volume_support, s_unif_clf,
                             s_X_clf, n_generated)[0]
            mv_clf += mv(axis_alpha, volume_support, s_unif_clf,
                             s_X_clf, n_generated)[0]

    em_clf /= averaging
    mv_clf /= averaging
    
    return em_clf, mv_clf

def get_em_mv_original(X, clf, alpha_min=0.9, alpha_max=0.999, 
                       n_generated=100000, t_max = 0.9):
    
    n_features = X.shape[1]
    
    lim_inf = X.min(axis=0)
    lim_sup = X.max(axis=0)
    volume_support = (lim_sup - lim_inf).prod()
    if volume_support == 0:
        volume_support = ((lim_sup - lim_inf) + 0.001).prod()
    print(volume_support)
    t = np.arange(0, 100 / volume_support, 0.01 / volume_support)
    axis_alpha = np.arange(alpha_min, alpha_max, 0.0001)
    unif = np.random.uniform(lim_inf, lim_sup,
                              size=(n_generated, n_features))

    clf.fit(X)
    s_X_clf = clf.decision_scores_ * -1
    s_unif_clf = clf.decision_function(unif) * -1
    

    em_clf = em(t, t_max, volume_support, s_unif_clf,
                s_X_clf, n_generated)[0]
    mv_clf = mv(axis_alpha, volume_support, s_unif_clf,
                s_X_clf, n_generated)[0]
    return em_clf, mv_clf