import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
import pickle
from datetime import datetime

print 'Naive Bayes using tfidf with fineline and selection'

with open('../data/features/pack_l1_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

with open('../data/feat_l1_union.pkl') as f:
    feat_list = pickle.load(f)

# filter the hand-made features
feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))

x_train = x_train[:, feat_list]
x_test = x_test[:, feat_list]

# record result & append
train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = MultinomialNB()
    clf.fit(x_train_sp, y_train_sp)
    pred = clf.predict_proba(x_test_sp)
    print log_loss(y_test_sp, pred)
