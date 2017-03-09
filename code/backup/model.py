import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pickle
import xgboost as xgb
from time import time


with open('./data/pack.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

# train test split
sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

best_iteration = []
best_score = []
iter_time = []
start_time = time()

for train_index, test_index in sss:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    dtrain = xgb.DMatrix(x_train_sp, y_train_sp)
    deval = xgb.DMatrix(x_test_sp, y_test_sp)
    watchlist = [(dtrain,'train'), (deval,'eval')]

    param = {'max_depth':10, 'eta':0.1, 'subsample':0.8, 'colsample_bytree':0.8, 'min_child_weight':20 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
    num_rounds = 20000

    clf = xgb.train(param, dtrain, num_rounds, watchlist, early_stopping_rounds=15)
    best_iteration.append(clf.best_iteration)
    best_score.append(clf.best_score)
    iter_time.append(time()-start_time)
    start_time = time()

for rnd, scr in zip(best_iteration, best_score):
    print 'Best iter_round is {}, best score is {}'.format(rnd, scr)



# dtrain = xgb.DMatrix(x_train, y_train)
# dtest = xgb.DMatrix(x_test)

# param = {'max_depth':10, 'eta':0.1, 'subsample':0.5, 'colsample_bytree':0.5, 'min_child_weight':5 , 'objective':'multi:softprob', 'eval_metric':'mlogloss', 'num_class':38}
# num_rounds = 300

# clf = xgb.train(param, dtrain, num_rounds)
# prob = clf.predict(dtest)

# df_result = pd.read_csv('./data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = prob
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('submission.csv', index=False)
