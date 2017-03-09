import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
import pickle
from datetime import datetime

# the func to solve mem problem by split x_test into slices
def divide_conquer(clf, x_test):
    len_test = x_test.shape[0]
    for i in range(0, len_test, 5000):
        if i == 0:
            pred = clf.predict_proba(x_test[i: min(i+5000, len_test), :])
        else:
            pred = np.vstack((pred, clf.predict_proba(x_test[i: min(i+5000, len_test), :])))
    return pred

######################################
print 'Nearest Neighbors using count with fineline, without hand feat, using euclid'

with open('../data/features/pack_count_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

x_train = x_train[:, 8:]
x_test = x_test[:, 8:]

# record result & append
train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='auto', metric='minkowski')
    clf.fit(x_train_sp, y_train_sp)
    pred = divide_conquer(clf, x_test_sp)
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='auto', metric='minkowski')
clf.fit(x_train, y_train)
pred = divide_conquer(clf, x_test)

with open('../data/models/knn_count_with_fineline_euc.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/knn_count_with_fineline_euc_submission.csv', index=False)

######################################
print 'Nearest Neighbors using count with fineline, without hand feat, using cosine'

with open('../data/features/pack_count_with_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

x_train = x_train[:, 8:]
x_test = x_test[:, 8:]

# record result & append
train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='brute', metric='cosine')
    clf.fit(x_train_sp, y_train_sp)
    pred = divide_conquer(clf, x_test_sp)
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='brute', metric='cosine')
clf.fit(x_train, y_train)
pred = divide_conquer(clf, x_test)

with open('../data/models/knn_count_with_fineline_cos.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/knn_count_with_fineline_cos_submission.csv', index=False)

######################################
print 'Nearest Neighbors using count with no fineline, with hand feat, using euclid'

with open('../data/features/pack_count_no_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

# record result & append
train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='auto', metric='minkowski')
    clf.fit(x_train_sp, y_train_sp)
    pred = divide_conquer(clf, x_test_sp)
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='auto', metric='minkowski')
clf.fit(x_train, y_train)
pred = divide_conquer(clf, x_test)

with open('../data/models/knn_count_no_fineline_euc.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/knn_count_no_fineline_euc_submission.csv', index=False)

######################################
print 'Nearest Neighbors using count with no fineline, with hand feat, using cosine'

with open('../data/features/pack_count_no_fineline.pkl') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

# record result & append
train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='brute', metric='cosine')
    clf.fit(x_train_sp, y_train_sp)
    pred = divide_conquer(clf, x_test_sp)
    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='brute', metric='cosine')
clf.fit(x_train, y_train)
pred = divide_conquer(clf, x_test)

with open('../data/models/knn_count_no_fineline_cos.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/knn_count_no_fineline_cos_submission.csv', index=False)







# with open('../data/features/pack_count_with_fineline.pkl') as f:
#     x_train, y_train, x_test = pickle.load(f)

# x_train = x_train[:, 8:]
# x_test = x_test[:, 8:]

# # with open('../data/feat_count_union.pkl') as f:
# #     feat_list = pickle.load(f)

# # # filter the hand-made features
# # feat_list = list(set(feat_list).difference({0,1,2,3,4,5,6,7}))

# # x_train = x_train[:, feat_list]
# # x_test = x_test[:, feat_list]

# sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# for train_index, test_index in sss:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

#     # clf = KNeighborsClassifier(n_neighbors=600, weights='uniform', algorithm='brute', metric='cosine')
#     clf = KNeighborsClassifier(n_neighbors=2, weights='uniform', algorithm='auto', metric='minkowski')
#     clf.fit(x_train_sp, y_train_sp)
#     pred1 = clf.predict_proba(x_test_sp[:5000])
#     pred2 = clf.predict_proba(x_test_sp[5000:10000])
#     pred3 = clf.predict_proba(x_test_sp[10000:15000])
#     pred4 = clf.predict_proba(x_test_sp[15000:20000])
#     pred5 = clf.predict_proba(x_test_sp[20000:25000])
#     pred6 = clf.predict_proba(x_test_sp[25000:])
#     pred = np.vstack([pred1, pred2, pred3, pred4, pred5, pred6])

#     score = log_loss(y_test_sp, pred)
#     print score