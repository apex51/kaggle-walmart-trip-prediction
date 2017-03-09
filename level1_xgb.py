import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
import pickle
import xgboost as xgb
from datetime import datetime

from scipy.sparse import hstack

###################################################

print '*'*20
print 'This is for xgboost with l1 feature'

with open('../data/features/pack_l1.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/features/upc_2_digit_l1_feat.pkl', 'rb') as f:
    upc_2_train, _, upc_2_test = pickle.load(f)

with open('../data/features/upc_3_digit_l1_feat.pkl', 'rb') as f:
    upc_3_train, _, upc_3_test = pickle.load(f)

x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

param = {'max_depth':5, 'eta':0.06, 'subsample':0.7, 'colsample_bytree':0.7, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
num_rounds = 858

# record result & append
train_result = np.array([])

# train test split
for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
    dtest = xgb.DMatrix(x_test_sp)

    clf = xgb.train(param, dtrain, num_rounds)
    pred = clf.predict(dtest)
    if not train_result.size:
        train_result = pred
    else:
        train_result = np.vstack((train_result, pred))

# train using the entire dataset then predict
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
clf = xgb.train(param, dtrain, num_rounds)
pred = clf.predict(dtest)

with open('../data/models/level1_xgb_l1.pkl', 'wb') as f:
    pickle.dump((train_result, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_xgb_l1.csv', index=False)

# ########################################################

print '*'*20
print 'This is for xgboost with count feature'

with open('../data/features/pack_count.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/features/upc_2_digit_count_feat.pkl', 'rb') as f:
    upc_2_train, _, upc_2_test = pickle.load(f)

with open('../data/features/upc_3_digit_count_feat.pkl', 'rb') as f:
    upc_3_train, _, upc_3_test = pickle.load(f)

x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

param = {'max_depth':5, 'eta':0.06, 'subsample':0.7, 'colsample_bytree':0.7, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
num_rounds = 858

# record result & append
train_result = np.array([])

# train test split
for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
    dtest = xgb.DMatrix(x_test_sp)

    clf = xgb.train(param, dtrain, num_rounds)
    pred = clf.predict(dtest)
    if not train_result.size:
        train_result = pred
    else:
        train_result = np.vstack((train_result, pred))

# train using the entire dataset then predict
dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(x_test)
clf = xgb.train(param, dtrain, num_rounds)
pred = clf.predict(dtest)

with open('../data/models/level1_xgb_count.pkl', 'wb') as f:
    pickle.dump((train_result, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_xgb_count.csv', index=False)

########################################################

# print '*'*20
# print 'This is for xgboost with count feature with no fineline feature'

# with open('../data/features/pack_count_no_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':8, 'eta':0.06, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38, 'silent':1}
# num_rounds = 332

# # record result & append
# train_result = np.array([])

# # train test split
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

# with open('../data/models/xgb_count_no_fineline.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/xgb_count_no_fineline.submission.csv', index=False)

########################################################

# print '*'*20
# print 'This is for xgboost with count feature with fineline feature'

# with open('../data/features/pack_count_with_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':5, 'eta':0.03, 'subsample':0.6, 'colsample_bytree':0.1, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38, 'silent':1}
# num_rounds = 3501

# # record result & append
# train_result = np.array([])

# # train test split
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

# with open('../data/models/xgb_count_with_feat_1.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/xgb_count_with_feat_submission_1.csv', index=False)

# ########################################################

# print '*'*20
# print 'This is for xgboost with count feature with fineline feature'

# with open('../data/features/pack_count_with_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':15, 'eta':0.06, 'subsample':0.8, 'colsample_bytree':0.4, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38, 'silent':1}
# num_rounds = 431

# # record result & append
# train_result = np.array([])

# # train test split
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

# with open('../data/models/xgb_count_with_feat_2.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/xgb_count_with_feat_submission_2.csv', index=False)

# ########################################################

# print '*'*20
# print 'This is for xgboost with l1 feature with fineline feature'

# with open('../data/features/pack_l1_with_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':7, 'eta':0.01, 'subsample':0.7, 'colsample_bytree':0.3, 'min_child_weight':4.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38, 'silent':1}
# num_rounds = 5076

# # record result & append
# train_result = np.array([])

# # train test split
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

# with open('../data/models/xgb_l1_with_feat_1.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/xgb_l1_with_feat_submission_1.csv', index=False)

# ########################################################

# print '*'*20
# print 'This is for xgboost with l1 feature with fineline feature'

# with open('../data/features/pack_l1_with_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# param = {'max_depth':17, 'eta':0.06, 'subsample':0.8, 'colsample_bytree':0.5, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38, 'silent':1}
# num_rounds = 361

# # record result & append
# train_result = np.array([])

# # train test split
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

# with open('../data/models/xgb_l1_with_feat_2.pkl', 'wb') as f:
#     pickle.dump((train_result, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/xgb_l1_with_feat_submission_2.csv', index=False)





















# import numpy as np
# import pandas as pd
# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import log_loss
# from sklearn.preprocessing import LabelEncoder
# import pickle
# import xgboost as xgb
# from datetime import datetime

# from hyperopt import hp
# from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

# from scipy.sparse import csr_matrix
# from scipy.sparse import hstack, vstack

# with open('../data/features/pack_l1.pkl', 'rb') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/features/upc_2_digit_l1_feat.pkl', 'rb') as f:
#     upc_2_train, _, upc_2_test = pickle.load(f)

# with open('../data/features/upc_3_digit_l1_feat.pkl', 'rb') as f:
#     upc_3_train, _, upc_3_test = pickle.load(f)

# x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
# x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

# with open('../data/feat_list.pkl', 'rb') as f:
#     feat_list = pickle.load(f)

# x_train_log = csr_matrix(np.log(1 + x_train[:, 20:].A))
# x_train = hstack((x_train[:, :20], x_train_log), 'csr')
# x_train_log = {}
# x_train = x_train[:, feat_list]
# x_test = x_test[:, feat_list]

###########################################################
# hyperopt code
###########################################################

# train test split
# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

# dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
# deval = xgb.DMatrix(x_test_sp, y_test_sp)
# watchlist = [(dtrain,'train'), (deval,'eval')]

# num_rounds = 200000

# def score(params):
#     clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
#     start = datetime.now()
#     score = clf.best_score
#     return {
#             'loss': score,
#             'status': STATUS_OK,
#             'eval_time': datetime.now() - start,
#             'best_iternum': clf.best_iteration}

# space = {
#         'max_depth': hp.quniform('max_depth', 1, 20, 1),
#         'eta': hp.quniform('eta', 0.01, 0.1, 0.01),
#         'subsample': hp.quniform('subsample', 0.1, 1.0, 0.1),
#         'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 1.0, 0.1),
#         'min_child_weight': hp.quniform('min_child_weight', 1, 5, 1),
#         'objective': 'multi:softprob',
#         'eval_metric': 'mlogloss',
#         'num_class': 38}

# trials = Trials()

# best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=15)

# print best

# with open('../log/level1_xgb_feats.pkl', 'wb') as f:
#     pickle.dump(trials, f)

###########################################################
# local validation code
###########################################################

# train test split
# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# best_iteration = []
# best_score = []
# iter_time = []
# # start_time = time()

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

#     dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
#     deval = xgb.DMatrix(x_test_sp, y_test_sp)
#     watchlist = [(dtrain,'train'), (deval,'eval')]

#     param = {'max_depth':8, 'eta':0.06, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':2, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
#     num_rounds = 200000

#     clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
#     best_iteration.append(clf.best_iteration)
#     best_score.append(clf.best_score)

# for rnd, scr in zip(best_iteration, best_score):
#     print 'Best iter_round is {}, best score is {}'.format(rnd, scr)

###########################################################
# submit code
###########################################################

# from datetime import datetime

# start = datetime.now()

# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)

# param = {'max_depth':5, 'eta':0.06, 'subsample':0.7, 'colsample_bytree':0.7, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 858

# clf = xgb.train(param, dtrain, num_rounds)
# prob = clf.predict(dtest)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = prob
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level1_xgb_l1_upc.csv', index=False)

# with open('../data/models/level1_xgb_l1_upc.pkl', 'wb') as f:
#     pickle.dump(clf, f)

# print datetime.now() - start

###########################################################
# parameter archive
###########################################################
# for ~5000 feats origin bow
# param = {'max_depth':5, 'eta':0.02, 'subsample':0.6, 'colsample_bytree':0.2, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 4600
