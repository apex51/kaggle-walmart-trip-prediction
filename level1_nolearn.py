import numpy as np
import pandas as pd
import pickle
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from lasagne.updates import nesterov_momentum, adagrad

from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
import theano

import random
from scipy.sparse import hstack, vstack

# #########################################################

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

x_train = x_train.toarray().astype(np.float32)
x_test = x_test.toarray().astype(np.float32)
y_train = y_train.astype(np.int32)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

num_features = x_train.shape[-1]

def float32(k):
    return np.cast['float32'](k)

# construct neural nets
def construct_net0():
    net0 = NeuralNet(layers=[('input', InputLayer),
                             ('dropoutf', DropoutLayer),
                             ('dense0', DenseLayer),
                             ('dropout', DropoutLayer),
                             ('dense1', DenseLayer),
                             ('dropout2', DropoutLayer),
                             ('output', DenseLayer)],
                     input_shape=(None, num_features),
                     dropoutf_p=0.02,
                     dense0_num_units=600,
                     dropout_p=0.75,
                     dense1_num_units=600,
                     dropout2_p=0.05,
                     output_num_units=38,
                     output_nonlinearity=softmax,
                     update=adagrad,
                     update_learning_rate=theano.shared(float32(0.008)),
                     eval_size=0.0,
                     verbose=1,
                     max_epochs=31)
    return net0

train_result = np.array([])
test_result = np.array([])

num_iter = 1

for i in range(num_iter):
    # use stratified 5 fold
    train_tmp = np.array([])

    for train_index, test_index in skf:
        x_train_sp = x_train[train_index]
        x_test_sp = x_train[test_index]
        y_train_sp = y_train[train_index]
        y_test_sp = y_train[test_index]

        clf = construct_net0()
        clf.fit(x_train_sp, y_train_sp)
        pred = clf.predict_proba(x_test_sp)

        if not train_tmp.size:
            train_tmp = pred
        else:
            train_tmp = np.vstack((train_tmp, pred))

    clf = construct_net0()
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)

    if not train_result.size:
        train_result = train_tmp
    else:
        train_result += train_tmp

    if not test_result.size:
        test_result = pred
    else:
        test_result += pred

train_result = train_result / num_iter
test_result = test_result / num_iter

with open('../data/models/level1_nn_count_nround.pkl', 'wb') as f:
    pickle.dump((train_result, test_result), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = test_result
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_nn_count_nround_submission.csv', index=False)

# # #########################################################

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

x_train = x_train.toarray().astype(np.float32)
x_test = x_test.toarray().astype(np.float32)
y_train = y_train.astype(np.int32)

with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)

num_features = x_train.shape[-1]

def float32(k):
    return np.cast['float32'](k)

# construct neural nets
def construct_net0():
    net0 = NeuralNet(layers=[('input', InputLayer),
                             ('dropoutf', DropoutLayer),
                             ('dense0', DenseLayer),
                             ('dropout', DropoutLayer),
                             ('dense1', DenseLayer),
                             ('dropout2', DropoutLayer),
                             ('output', DenseLayer)],
                     input_shape=(None, num_features),
                     dropoutf_p=0.025,
                     dense0_num_units=600,
                     dropout_p=0.75,
                     dense1_num_units=600,
                     dropout2_p=0.05,
                     output_num_units=38,
                     output_nonlinearity=softmax,
                     update=adagrad,
                     update_learning_rate=theano.shared(float32(0.01)),
                     eval_size=.0,
                     verbose=1,
                     max_epochs=43)
    return net0

train_result = np.array([])
test_result = np.array([])

num_iter = 1

for i in range(num_iter):
    # use stratified 5 fold
    train_tmp = np.array([])

    for train_index, test_index in skf:
        x_train_sp = x_train[train_index]
        x_test_sp = x_train[test_index]
        y_train_sp = y_train[train_index]
        y_test_sp = y_train[test_index]

        clf = construct_net0()
        clf.fit(x_train_sp, y_train_sp)
        pred = clf.predict_proba(x_test_sp)

        if not train_tmp.size:
            train_tmp = pred
        else:
            train_tmp = np.vstack((train_tmp, pred))

    clf = construct_net0()
    clf.fit(x_train, y_train)
    pred = clf.predict_proba(x_test)

    if not train_result.size:
        train_result = train_tmp
    else:
        train_result += train_tmp

    if not test_result.size:
        test_result = pred
    else:
        test_result += pred

train_result = train_result / num_iter
test_result = test_result / num_iter

with open('../data/models/level1_nn_l1_nround.pkl', 'wb') as f:
    pickle.dump((train_result, test_result), f)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = test_result
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../data/models/level1_nn_l1_nround_submission.csv', index=False)



######################################

#########################################################

# import random
# from scipy.sparse import hstack, vstack

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

# x_train = x_train.toarray().astype(np.float32)
# x_test = x_test.toarray().astype(np.float32)
# y_train = y_train.astype(np.int32)

# num_features = x_train.shape[-1]

# class EarlyStopping(object):
#     def __init__(self, patience=100):
#         self.patience = patience
#         self.best_valid = np.inf
#         self.best_valid_epoch = 0
#         self.best_weights = None

#     def __call__(self, nn, train_history):
#         current_valid = train_history[-1]['valid_loss']
#         current_epoch = train_history[-1]['epoch']
#         if current_valid < self.best_valid:
#             self.best_valid = current_valid
#             self.best_valid_epoch = current_epoch
#             self.best_weights = nn.get_all_params_values()
#         elif self.best_valid_epoch + self.patience < current_epoch:
#             print("Early stopping.")
#             print("Best valid loss was {:.6f} at epoch {}.".format(
#                 self.best_valid, self.best_valid_epoch))
#             nn.load_params_from(self.best_weights)
#             raise StopIteration()

# def float32(k):
#     return np.cast['float32'](k)

# # construct neural nets
# layers0 = [('input', InputLayer),
#            ('dropoutf', DropoutLayer),
#            ('dense0', DenseLayer),
#            ('dropout', DropoutLayer),
#            ('dense1', DenseLayer),
#            ('dropout2', DropoutLayer),
#            ('output', DenseLayer)]

# net0 = NeuralNet(layers=layers0,
#                  input_shape=(None, num_features),
#                  dropoutf_p=0.025,
#                  dense0_num_units=600,
#                  dropout_p=0.75,
#                  dense1_num_units=600,
#                  dropout2_p=0.05,
#                  output_num_units=38,
#                  output_nonlinearity=softmax,
#                  update=adagrad,
#                  update_learning_rate=theano.shared(float32(0.01)),
#                  eval_size=0.3,
#                  verbose=1,
#                  max_epochs=100)

# net0.fit(x_train, y_train)
# # pred = net0.predict_proba(x_test)
