import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import log_loss
import pickle
import xgboost as xgb

with open('../data/features/pack_count.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

###########################################################
# local validation code
###########################################################

# train test split
skf = StratifiedKFold(y_train, n_folds=5, shuffle=True, random_state=22)

feats_matrix = []

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
    deval = xgb.DMatrix(x_test_sp, y_test_sp)
    watchlist = [(dtrain,'train'), (deval,'eval')]

    param = {'max_depth':8, 'eta':0.06, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
    num_rounds = 200000

    clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
    f_importance = clf.get_fscore()
    feats_matrix.append(f_importance)

with open('../data/feats_matrix_count.pkl', 'wb') as f:
    pickle.dump(feats_matrix, f)

print 'This is for count data with feat'

