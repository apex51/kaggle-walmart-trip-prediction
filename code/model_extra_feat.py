import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
import pickle
from datetime import datetime
from scipy.sparse import vstack

with open('../data/features/pack_tfidf_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

# with open('../data/feat_count_union.pkl') as f:
#     feat_list = pickle.load(f)

# filter the hand-made features
# feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))
feat_list = range(10)

x_train = x_train[:, feat_list]
x_test = x_test[:, feat_list]

# record result & append
train_tmp = np.array([])

for _, test_index in skf:
    x_test_sp = x_train[test_index]
    
    if not train_tmp.size:
        train_tmp = x_test_sp
    else:
        train_tmp = vstack((train_tmp, x_test_sp), 'csr')

with open('../data/models/extra_20_feats.pkl', 'wb') as f:
    pickle.dump((train_tmp, x_test), f)