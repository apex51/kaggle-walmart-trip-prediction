import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb
from datetime import datetime

from hyperopt import hp
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

###############################################
# load train set, test set, train y from pickle
###############################################
# nb count with fineline
with open('../data/models/level2_nn.pkl') as f:
    train_1, test_1 = pickle.load(f)
    train_1 = train_1.astype(np.float32)
    test_1 = test_1.astype(np.float32)

# nb tfidf with fineline
with open('../data/models/level2_xgb.pkl') as f:
    train_2, test_2 = pickle.load(f)
    train_2 = train_2.astype(np.float32)
    test_2 = test_2.astype(np.float32)

# train_y with stratified 5 fold and over sampling
with open('../data/features/pack_count_no_fineline.pkl') as f:
    _0, y_train, _1 = pickle.load(f)
# this skf is for level_1
with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)
# generate y_train for level_2
y_tmp = np.array([])
for _, test_index in skf:
    y_tmp = np.append(y_tmp, y_train[test_index])
y_train = y_tmp
# then grnerate y_train for level 3
with open('../data/stratified_kfold_level2.pkl') as f:
    skf_level2 = pickle.load(f)
y_tmp = np.array([])
for _, test_index in skf_level2:
    y_tmp = np.append(y_tmp, y_train[test_index])

# get ready the x_train, y_train and x_test
x_train = np.hstack((train_1, train_2))
x_test = np.hstack((test_1, test_2))
y_train = y_tmp

# # log scaler
# x_train = np.log(x_train + 1)
# x_test = np.log(x_test + 1)

########################################################

# print '*'*20
# print 'This is for xgboost on level 2'

# with open('../data/stratified_kfold_level2.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':3.0, 'eta':0.02, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':3.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 749

# # record result & append
# train_result = np.array([])

# # 4 stratified fold
# for train_index, test_index in skf:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

#     dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
#     dtest = xgb.DMatrix(x_test_sp)

#     clf = xgb.train(param, dtrain, num_rounds)
#     pred = clf.predict(dtest)
#     if not train_result.size:
#         train_result = pred
#     else:
#         train_result = np.vstack((train_result, pred))

# # train using the entire dataset then predict
# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)
# clf = xgb.train(param, dtrain, num_rounds)
# pred = clf.predict(dtest)

# with open('../data/models/level2_xgb.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level2_xgb_submission.csv', index=False)






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

#     dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
#     deval = xgb.DMatrix(x_test_sp, y_test_sp)
#     watchlist = [(dtrain,'train'), (deval,'eval')]

#     param = {'max_depth':3.0, 'eta':0.02, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':3.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
#     # num_rounds = 749
#     num_rounds = 100000

#     clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)

###########################################################
# submit code
###########################################################

# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)

# param = {'max_depth':3.0, 'eta':0.02, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':3.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 590

# clf = xgb.train(param, dtrain, num_rounds)
# prob = clf.predict(dtest)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = prob
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level3_xgb_submission.csv', index=False)
