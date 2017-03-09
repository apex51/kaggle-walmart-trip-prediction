import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.cross_validation import StratifiedShuffleSplit
from scipy.sparse import hstack, vstack

###########################################
# count features with fineline

# load from file
with open('../data/features/pack_l1.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/features/upc_2_digit_l1_feat.pkl', 'rb') as f:
    upc_2_train, _, upc_2_test = pickle.load(f)

with open('../data/features/upc_3_digit_l1_feat.pkl', 'rb') as f:
    upc_3_train, _, upc_3_test = pickle.load(f)

x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

with open('../data/features/select_upc_l1_list.pkl') as f:
    feat_index = pickle.load(f)

x_train = x_train[:, feat_index]
x_test = x_test[:, feat_index]

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(x_train_sp, y_train_sp)
    pred = clf_isotonic.predict_proba(x_test_sp)

    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(x_train, y_train)
pred = clf_isotonic.predict_proba(x_test)

with open('../data/models/level1_rf_l1.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_rf_l1_submission.csv', index=False)


# ###########################################
# l1 features with fineline

# load from file
with open('../data/features/pack_count.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/features/upc_2_digit_count_feat.pkl', 'rb') as f:
    upc_2_train, _, upc_2_test = pickle.load(f)

with open('../data/features/upc_3_digit_count_feat.pkl', 'rb') as f:
    upc_3_train, _, upc_3_test = pickle.load(f)

x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

with open('../data/features/select_upc_count_list.pkl') as f:
    feat_index = pickle.load(f)

x_train = x_train[:, feat_index]
x_test = x_test[:, feat_index]

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

train_tmp = np.array([])

for train_index, test_index in skf:
    x_train_sp = x_train[train_index]
    x_test_sp = x_train[test_index]
    y_train_sp = y_train[train_index]
    y_test_sp = y_train[test_index]

    clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
    clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
    clf_isotonic.fit(x_train_sp, y_train_sp)
    pred = clf_isotonic.predict_proba(x_test_sp)

    if not train_tmp.size:
        train_tmp = pred
    else:
        train_tmp = np.vstack((train_tmp, pred))

clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
clf_isotonic.fit(x_train, y_train)
pred = clf_isotonic.predict_proba(x_test)

with open('../data/models/level1_rf_count.pkl', 'wb') as f:
    pickle.dump((train_tmp, pred), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_rf_count_submission.csv', index=False)

# ###########################################
# # features with no fineline

# with open('../data/features/pack_count_no_fineline.pkl', 'rb') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/stratified_kfold.pkl') as f:
#     skf = pickle.load(f)

# train_tmp = np.array([])

# for train_index, test_index in skf:
#     x_train_sp = x_train[train_index]
#     x_test_sp = x_train[test_index]
#     y_train_sp = y_train[train_index]
#     y_test_sp = y_train[test_index]

#     clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
#     clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
#     clf_isotonic.fit(x_train_sp, y_train_sp)
#     pred = clf_isotonic.predict_proba(x_test_sp)

#     if not train_tmp.size:
#         train_tmp = pred
#     else:
#         train_tmp = np.vstack((train_tmp, pred))

# clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
# clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
# clf_isotonic.fit(x_train, y_train)
# pred = clf_isotonic.predict_proba(x_test)

# with open('../data/models/rf_count_no_fineline_calib.pkl', 'wb') as f:
#     pickle.dump((train_tmp, pred), f)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/rf_count_no_fineline_calib_submission.csv', index=False)


#####################################

# # load from file
# with open('../data/features/pack_l1.pkl', 'rb') as f:
#     x_train, y_train, x_test = pickle.load(f)

# with open('../data/features/upc_2_digit_l1_feat.pkl', 'rb') as f:
#     upc_2_train, _, upc_2_test = pickle.load(f)

# with open('../data/features/upc_3_digit_l1_feat.pkl', 'rb') as f:
#     upc_3_train, _, upc_3_test = pickle.load(f)

# x_train = hstack((x_train, upc_2_train, upc_3_train), 'csr')
# x_test = hstack((x_test, upc_2_test, upc_3_test), 'csr')

# with open('../data/features/select_upc_l1_list.pkl') as f:
#     feat_index = pickle.load(f)

# x_train = x_train[:, feat_index]
# x_test = x_test[:, feat_index]

# # sss = StratifiedShuffleSplit(y_train, n_iter=1, test_size=0.3, random_state=22)

# # for train_index, test_index in sss:
# #     x_train_sp = x_train[train_index]
# #     x_test_sp = x_train[test_index]
# #     y_train_sp = y_train[train_index]
# #     y_test_sp = y_train[test_index]

# clf = RandomForestClassifier(n_estimators=200, n_jobs=-1)
# clf_isotonic = CalibratedClassifierCV(clf, cv=2, method='isotonic')
# clf_isotonic.fit(x_train, y_train)
# pred = clf_isotonic.predict_proba(x_test)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = pred
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level1_rf_l1.csv', index=False)

# # score = log_loss(y_test_sp, pred)
# # print score