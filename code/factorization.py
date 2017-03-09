import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
import pickle
import xgboost as xgb
from datetime import datetime
from scipy.sparse import hstack, vstack
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

with open('../data/features/pack_count_no_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

x_train_hand = x_train[:, :8]
x_train_count = x_train[:, 8:142]
x_test_hand = x_test[:, :8]
x_test_count = x_test[:, 8:142]

svd = TruncatedSVD(n_components=40, algorithm='arpack', random_state=42)
svd.fit(vstack((x_train_count, x_test_count), 'csr'))
x_train_count = svd.transform(x_train_count)
x_test_count = svd.transform(x_test_count)

x_train = hstack((x_train_hand, x_train_count), 'csr')
x_test = hstack((x_test_hand, x_test_count), 'csr')

############################################ xgb
# # train test split
# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

#     dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
#     deval = xgb.DMatrix(x_test_sp, y_test_sp)
#     watchlist = [(dtrain,'train'), (deval,'eval')]

#     param = {'max_depth':8, 'eta':0.06, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':2.0 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
#     num_rounds = 200000

#     clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
#     best_iteration.append(clf.best_iteration)
#     best_score.append(clf.best_score)

############################################ rf
# # train test split
sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

for train_index, test_index in sss:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(x_train_sp, y_train_sp)
    pred = clf_isotonic.predict_proba(x_test_sp)
    # clf.fit(x_train_sp, y_train_sp)
    # pred = clf.predict_proba(x_test_sp)    
    print log_loss(y_test_sp, pred)

