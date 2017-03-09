import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
import pickle

###############################################
# load train set, test set, train y from pickle
###############################################
# # nb count with fineline
# with open('../data/models/4fold/level2_nn.pkl') as f:
#     train_1, test_1 = pickle.load(f)
#     train_1 = train_1.astype(np.float32)
#     test_1 = test_1.astype(np.float32)

# # nb tfidf with fineline
# with open('../data/models/4fold/level2_xgb.pkl') as f:
#     train_2, test_2 = pickle.load(f)
#     train_2 = train_2.astype(np.float32)
#     test_2 = test_2.astype(np.float32)

# # train_y with stratified 5 fold and over sampling
# with open('../data/features/pack_count_no_fineline.pkl') as f:
#     _0, y_train, _1 = pickle.load(f)
# # this skf is for level_1
# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)
# # generate y_train for level_2
# y_tmp = np.array([])
# for _, test_index in skf:
#     y_tmp = np.append(y_tmp, y_train[test_index])
# y_train = y_tmp
# # then grnerate y_train for level 3
# with open('../data/stratified_kfold_level2.pkl') as f:
#     skf_level2 = pickle.load(f)
# y_tmp = np.array([])
# for _, test_index in skf_level2:
#     y_tmp = np.append(y_tmp, y_train[test_index])

# # get ready the x_train, y_train and x_test
# x_train = np.hstack((train_1, train_2))
# x_test = np.hstack((test_1, test_2))
# y_train = y_tmp

# def constraint(w, *args):
#     return min(w) - .0

# def log_score(w, pred_0, pred_1, target):
#     pred = pred_0 * w[0] + pred_1 * w[1]
#     return log_loss(target, pred)

# w0 = [1.0, 1.0]
# weights = fmin_cobyla(log_score, w0, args=(train_1, train_2, y_train), cons=[constraint], rhoend=1e-5)

# pred = test_1 * weights[0] + test_2 * weights[1]

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level3_cobyla_submission.csv', index=False)

mode = 0 # 0:arithmetic 1:geometric
w0 = 0.66 # xgb coef
w1 = 1 - w0

# pred0 = pd.read_csv('../data/models/4fold/level2_xgb_submission.csv')
# pred1 = pd.read_csv('../data/models/4fold/level2_nn_submission.csv')
pred0 = pd.read_csv('../data/models/level2_xgb_upc_submission.csv')
pred1 = pd.read_csv('../data/models/level2_nn_upc_submission_40round.csv')


pred0 = pred0.iloc[:, 1:]
pred1 = pred1.iloc[:, 1:]

if mode == 0:
    pred = pred0 * w0 + pred1 * w1 # arithmetic average
else:
    pred = (pred0 ** w0) * (pred1 ** w1)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level3_average_submission_{}_{}_{}_1227.csv'.format(mode, w0, w1), index=False)

