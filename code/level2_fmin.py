import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
import pickle
from scipy.optimize import fmin_cobyla

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
    train_8 = train_8.astype(np.float32)
    test_8 = test_8.astype(np.float32)

# xgb count with feat
with open('../data/models/xgb_count_with_feat.pkl') as f:
    train_9, test_9 = pickle.load(f)
    train_9 = train_9.astype(np.float32)
    test_9 = test_9.astype(np.float32)

# xgb l1 with feat
with open('../data/models/xgb_l1_with_feat.pkl') as f:
    train_10, test_10 = pickle.load(f)
    train_10 = train_10.astype(np.float32)
    test_10 = test_10.astype(np.float32)

# knn count no fineline cos
with open('../data/models/knn_count_no_fineline_cos.pkl') as f:
    train_11, test_11 = pickle.load(f)
    train_11 = train_11.astype(np.float32)
    test_11 = test_11.astype(np.float32)

# knn count no fineline euc
with open('../data/models/knn_count_no_fineline_euc.pkl') as f:
    train_12, test_12 = pickle.load(f)
    train_12 = train_12.astype(np.float32)
    test_12 = test_12.astype(np.float32)

# knn count with fineline cos
with open('../data/models/knn_count_with_fineline_cos.pkl') as f:
    train_13, test_13 = pickle.load(f)
    train_13 = train_13.astype(np.float32)
    test_13 = test_13.astype(np.float32)

# knn count with fineline euc
with open('../data/models/knn_count_with_fineline_euc.pkl') as f:
    train_14, test_14 = pickle.load(f)
    train_14 = train_14.astype(np.float32)
    test_14 = test_14.astype(np.float32)

# xgb count with fineline use 3501 iters
with open('../data/models/xgb_count_with_feat_1.pkl') as f:
    train_15, test_15 = pickle.load(f)
    train_15 = train_15.astype(np.float32)
    test_15 = test_15.astype(np.float32)

# xgb count with fineline use 431 iters
with open('../data/models/xgb_count_with_feat_2.pkl') as f:
    train_16, test_16 = pickle.load(f)
    train_16 = train_16.astype(np.float32)
    test_16 = test_16.astype(np.float32)

# xgb count with fineline use 5076 iters
with open('../data/models/xgb_l1_with_feat_1.pkl') as f:
    train_17, test_17 = pickle.load(f)
    train_17 = train_17.astype(np.float32)
    test_17 = test_17.astype(np.float32)

# xgb count with fineline use 361 iters
with open('../data/models/xgb_l1_with_feat_2.pkl') as f:
    train_18, test_18 = pickle.load(f)
    train_18 = train_18.astype(np.float32)
    test_18 = test_18.astype(np.float32)

# train_y with stratified 5 fold and over sampling
with open('../data/features/pack_count_no_fineline.pkl') as f:
    _0, y_train, _1 = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

y_tmp = np.array([])
for _, test_index in skf:
    y_tmp = np.append(y_tmp, y_train[test_index])
y_train = y_tmp

###############################################
# use cobyla to calculate weight
###############################################

def constraint(w, *args):
    return min(w) - .0

def log_score(w, train_data, target):
    pred = np.zeros(train_data[-1].shape)
    for i in range(len(train_data)):
        pred += train_data[i] * w[i]
    return log_loss(target, pred)

train_data = (train_1, train_2, train_3, train_4, train_5, train_6, train_7, train_8, train_9, train_10, train_11, train_12, train_13, train_14, train_15, train_16, train_17, train_18)
w0 = [1.0] * 18
weights = fmin_cobyla(log_score, w0, args=(train_data, y_train), cons=[constraint], rhoend=1e-5)

pred = test_1 * weights[0] + test_2 * weights[1]

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level3_cobyla_submission.csv', index=False)


