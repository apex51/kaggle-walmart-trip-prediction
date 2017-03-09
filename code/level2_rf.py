import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb
from datetime import datetime

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from scipy.sparse import vstack, hstack

###############################################
# load train set, test set, train y from pickle
###############################################
# nb count with fineline
with open('../data/models/nb_count_with_feat.pkl') as f:
    train_1, test_1 = pickle.load(f)
    train_1 = train_1.astype(np.float32)
    test_1 = test_1.astype(np.float32)

# nb tfidf with fineline
with open('../data/models/nb_tfidf_with_feat.pkl') as f:
    train_2, test_2 = pickle.load(f)
    train_2 = train_2.astype(np.float32)
    test_2 = test_2.astype(np.float32)

# nn count with feat * 30
with open('../data/models/nn_count_with_feat_30.pkl') as f:
    train_3, test_3 = pickle.load(f)

# nn l1 with feat * 30
with open('../data/models/nn_l1_with_feat_30.pkl') as f:
    train_4, test_4 = pickle.load(f)

# rf count no fileine calib
with open('../data/models/rf_count_no_fineline_calib.pkl') as f:
    train_5, test_5 = pickle.load(f)
    train_5 = train_5.astype(np.float32)
    test_5 = test_5.astype(np.float32)

# rf count with fineline calib
with open('../data/models/rf_count_with_fineline_calib.pkl') as f:
    train_6, test_6 = pickle.load(f)
    train_6 = train_6.astype(np.float32)
    test_6 = test_6.astype(np.float32)

# rf l1 with fineline calib
with open('../data/models/rf_l1_with_fineline_calib.pkl') as f:
    train_7, test_7 = pickle.load(f)
    train_7 = train_7.astype(np.float32)
    test_7 = test_7.astype(np.float32)

# xgb count no fineline
with open('../data/models/xgb_count_no_fineline.pkl') as f:
    train_8, test_8 = pickle.load(f)

# xgb count with feat
with open('../data/models/xgb_count_with_feat.pkl') as f:
    train_9, test_9 = pickle.load(f)

# xgb l1 with feat
with open('../data/models/xgb_l1_with_feat.pkl') as f:
    train_10, test_10 = pickle.load(f)

# knn count no fineline cos
with open('../data/models/knn_count_no_fineline_cos.pkl') as f:
    train_11, test_11 = pickle.load(f)

# knn count no fineline euc
with open('../data/models/knn_count_no_fineline_euc.pkl') as f:
    train_12, test_12 = pickle.load(f)

# knn count with fineline cos
with open('../data/models/knn_count_with_fineline_cos.pkl') as f:
    train_13, test_13 = pickle.load(f)

# knn count with fineline euc
with open('../data/models/knn_count_with_fineline_euc.pkl') as f:
    train_14, test_14 = pickle.load(f)

# xgb count with fineline use 3501 iters
with open('../data/models/xgb_count_with_feat_1.pkl') as f:
    train_15, test_15 = pickle.load(f)

# xgb count with fineline use 431 iters
with open('../data/models/xgb_count_with_feat_2.pkl') as f:
    train_16, test_16 = pickle.load(f)

# xgb count with fineline use 5076 iters
with open('../data/models/xgb_l1_with_feat_1.pkl') as f:
    train_17, test_17 = pickle.load(f)

# xgb count with fineline use 361 iters
with open('../data/models/xgb_l1_with_feat_2.pkl') as f:
    train_18, test_18 = pickle.load(f)

# train_y with stratified 5 fold and over sampling
with open('../data/features/pack_count_no_fineline.pkl') as f:
    _0, y_train, _1 = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

y_tmp = np.array([])
for _, test_index in skf:
    y_tmp = np.append(y_tmp, y_train[test_index])

# get ready the x_train, y_train and x_test
x_train = np.hstack((train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18))
x_test = np.hstack((test_1, test_2, test_3, test_4, test_5, test_6, test_7, test_8, test_9, test_10, test_11, test_12, test_13, test_14, test_15, test_16, test_17, test_18))
y_train = y_tmp

# over sampling for the 10-fold thing
x_train = vstack((x_train, x_train[y_train == 8]), 'csr')
y_train = np.append(y_train, y_train[y_train == 8])

########################################################

with open('../data/stratified_kfold_level2.pkl') as f:
    skf = pickle.load(f)

# record result & append
train_result = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = RandomForestClassifier(n_estimators=600, n_jobs=-1)
    clf.fit(x_train_sp, y_train_sp)
    pred = clf.predict_proba(x_test_sp)

    if not train_result.size:
        train_result = pred
    else:
        train_result = np.vstack((train_result, pred))

# train using the entire dataset then predict
clf = RandomForestClassifier(n_estimators=600, n_jobs=-1)
clf.fit(x_train, y_train)
pred = clf.predict_proba(x_test)

with open('../data/models/level2_rf.pkl', 'wb') as f:
    pickle.dump((train_result, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level2_rf_submission.csv', index=False)

###############################################
# xgb
###############################################

# # train test split
# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=66)

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

# clf = RandomForestClassifier(n_estimators=100, max_features=50, n_jobs=-1)
# # clf.fit(x_train_sp, y_train_sp)
# # pred = clf.predict_proba(x_test_sp)
# # print log_loss(y_test_sp, pred)
# clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
# clf_isotonic.fit(x_train_sp, y_train_sp)
# pred = clf_isotonic.predict_proba(x_test_sp)
# print log_loss(y_test_sp, pred)
