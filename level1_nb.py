import numpy as np
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import log_loss
import pickle
from datetime import datetime

print 'Naive Bayes using tfidf with fineline and selection'

with open('../data/features/pack_tfidf_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

# with open('../data/feat_count_union.pkl') as f:
#     feat_list = pickle.load(f)

# filter the hand-made features
# feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))
feat_list = range(20, 184) + [1, 3, 8, 9, 11, 18, 19]

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
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = MultinomialNB()
clf.fit(x_train, y_train)
pred = clf.predict_proba(x_test)

with open('../data/models/nb_tfidf_with_feat.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/nb_tfidf_with_feat_submission.csv', index=False)

#########################################################

print 'Naive Bayes using count with fineline and selection'

with open('../data/features/pack_count_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

# with open('../data/feat_count_union.pkl') as f:
#     feat_list = pickle.load(f)

# filter the hand-made features
# feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))
feat_list = range(20, 184) + [1, 3, 8, 9, 11, 18, 19]

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
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = MultinomialNB()
clf.fit(x_train, y_train)
pred = clf.predict_proba(x_test)

with open('../data/models/nb_count_with_feat.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/nb_count_with_feat_submission.csv', index=False)







# import numpy as np
# import pandas as pd
# from sklearn.cross_validation import train_test_split
# from sklearn.cross_validation import StratifiedShuffleSplit
# from sklearn.metrics import log_loss
# from sklearn.preprocessing import LabelEncoder
# import pickle
# from datetime import datetime
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.metrics import log_loss

# import random

# from scipy.sparse import hstack, vstack


# # ########################
# # hand feat: 10 + 10
# # depart feat: 164
# # fine feat: 5323 + 3936
# # ########################
# # using feat bagging
# # ########################

# with open('../data/features/pack_count_with_fineline.pkl', 'rb') as f:
#     x_train, y_train, x_test = pickle.load(f)

# # average_buy = x_train[:, 1] / (x_train[:, 3].A + 1)
# # average_buy_log = np.log(average_buy + 1)
# # count_buy_log = np.log(x_train[:, 1].A +1)
# # num_trans = x_train[:, 1] + x_train[:, 2]
# # num_trans_log = np.log(num_trans.A + 1)
# # x_train = hstack((x_train, average_buy, count_buy_log, average_buy_log, num_trans, num_trans_log), 'csr')

# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# for train_index, test_index in sss:
#     x_train_sp_origin = x_train[train_index]
#     x_test_sp_origin = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

# result = np.array([])
# num_iter = 1

# for i in range(num_iter):
#     # rand_index = random.sample(xrange(172, x_train.shape[-1]), 100)
#     # feat_index = rand_index + range(1, 172)
#     # feat_index = range(20, 184) + [1, 3, -5, -2, -3, -1, -4]
#     feat_index = range(20, 184) + [1, 3, 8, 9, 18, 19, 11]
#     x_train_sp = x_train_sp_origin[:, feat_index]
#     x_test_sp = x_test_sp_origin[:, feat_index]

#     clf = MultinomialNB()
#     clf.fit(x_train_sp, y_train_sp)
#     pred = clf.predict_proba(x_test_sp)

#     if not result.size:
#         result = pred
#     else:
#         result += pred

# result = result / num_iter

# score = log_loss(y_test_sp, result)
# print score

# ########################
# using xgb selected feat
# ########################

# with open('../data/features/pack_count_with_fineline.pkl', 'rb') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/feat_count_union.pkl') as f:
#     feat_list = pickle.load(f)

# # filter the hand-made features
# feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))

# x_train = x_train[:, feat_list]
# x_test = x_test[:, feat_list]

# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

# clf = MultinomialNB()
# clf.fit(x_train_sp, y_train_sp)
# pred = clf.predict_proba(x_test_sp)

# score = log_loss(y_test_sp, result)
# print score