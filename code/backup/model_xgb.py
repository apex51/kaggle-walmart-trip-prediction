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


with open('../data/pack.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

# with open('../data/feat_list.pkl', 'rb') as f:
#     feat_list = pickle.load(f)

# x_train = x_train[:, feat_list]
# x_test = x_test[:, feat_list]

###########################################################
# hyperopt code
###########################################################

# train test split
sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

for train_index, test_index in sss:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
deval = xgb.DMatrix(x_test_sp, y_test_sp)
watchlist = [(dtrain,'train'), (deval,'eval')]

num_rounds = 200000

def score(params):
    clf = xgb.train(params, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
    start = datetime.now()
    score = clf.best_score
    return {
            'loss': score,
            'status': STATUS_OK,
            'eval_time': datetime.now() - start,
            'best_iternum': clf.best_iteration}

space = {
        'max_depth': hp.quniform('max_depth', 1, 20, 1),
        'eta': 0.01,
        'subsample': hp.quniform('subsample', 0.1, 0.80, 0.1),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.1, 0.8, 0.1),
        'min_child_weight': 5.0,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'num_class': 38}

trials = Trials()

best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=10)

print best

###########################################################
# local validation code
###########################################################

# # train test split
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

#     # param = {'max_depth':15, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.3, 'min_child_weight':20 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
#     # num_rounds = 20000


#     param = {'max_depth':8, 'eta':0.01, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
#     num_rounds = 200000

#     clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
#     best_iteration.append(clf.best_iteration)
#     best_score.append(clf.best_score)
#     # iter_time.append(time()-start_time)
#     # start_time = time()

# for rnd, scr in zip(best_iteration, best_score):
#     print 'Best iter_round is {}, best score is {}'.format(rnd, scr)

###########################################################
# submit code
###########################################################

# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)

# param = {'max_depth':8, 'eta':0.08, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':1.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 515

# clf = xgb.train(param, dtrain, num_rounds)
# prob = clf.predict(dtest)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = prob
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../submission.csv', index=False)

###########################################################
# parameter archive
###########################################################
# for ~5000 feats origin bow
# param = {'max_depth':5, 'eta':0.02, 'subsample':0.6, 'colsample_bytree':0.2, 'min_child_weight':2.0, 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 4600
