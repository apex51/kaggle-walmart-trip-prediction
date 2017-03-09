import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pickle
from datetime import datetime
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss

import random

# hand feat: 8
# depart feat: 164
# fine feat: 5323 + 3936

with open('../data/features/pack_count_with_fineline.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

for train_index, test_index in sss:
    x_train_sp_origin = x_train[train_index]
    x_test_sp_origin = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

result = np.array([])
num_iter = 10

for i in range(num_iter):
    rand_index = random.sample(xrange(172, x_train.shape[-1]), 1000)
    feat_index = rand_index + range(172)

    x_train_sp = x_train_sp_origin[:, feat_index]
    x_test_sp = x_test_sp_origin[:, feat_index]

    clf = MultinomialNB()
    clf.fit(x_train_sp, y_train_sp)
    pred = clf.predict_proba(x_test_sp)

    if not result.size():
        result = pred
    else:
        result += pred

result = result / num_iter

score = log_loss(y_test_sp, result)
print score




