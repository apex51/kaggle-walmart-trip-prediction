import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import log_loss
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

###############################################
# load train set, test set, train y from pickle
###############################################
# nb count with fineline
with open('../data/models/level2_nn.pkl') as f:
    train_1, test_1 = pickle.load(f)
    train_1 = train_1.astype(np.float32)
    test_1 = test_1.astype(np.float32)

# nb tfidf with fineline
with open('../data/models/level2_xgb.pkl') as f:
    train_2, test_2 = pickle.load(f)
    train_2 = train_2.astype(np.float32)
    test_2 = test_2.astype(np.float32)

# train_y with stratified 5 fold and over sampling
with open('../data/features/pack_count_no_fineline.pkl') as f:
    _0, y_train, _1 = pickle.load(f)
# this skf is for level_1
with open('../data/stratified_kfold.pkl') as f:
    skf = pickle.load(f)
# generate y_train for level_2
y_tmp = np.array([])
for _, test_index in skf:
    y_tmp = np.append(y_tmp, y_train[test_index])
y_train = y_tmp
# then grnerate y_train for level 3
with open('../data/stratified_kfold_level2.pkl') as f:
    skf_level2 = pickle.load(f)
y_tmp = np.array([])
for _, test_index in skf_level2:
    y_tmp = np.append(y_tmp, y_train[test_index])

# get ready the x_train, y_train and x_test
x_train = np.hstack((train_1, train_2))
x_test = np.hstack((test_1, test_2))
y_train = y_tmp

# # log scaler
x_train = np.log(x_train + 1)
x_test = np.log(x_test + 1)

###############################################
# nolearn
###############################################

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def float32(k):
    return np.cast['float32'](k)

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.int32)
num_features = x_train.shape[-1]

# construct neural nets
layers0 = [('input', InputLayer),
           ('dropoutf', DropoutLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           # ('dense1', DenseLayer),
           # ('dropout2', DropoutLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 input_shape=(None, num_features),
                 dropoutf_p=0.2,
                 dense0_num_units=600,
                 dropout_p=0.2,
                 # dense1_num_units=200,
                 # dropout2_p=0.2,
                 output_num_units=38,
                 output_nonlinearity=softmax,
                 # on_epoch_finished=[
                 #    EarlyStopping(patience=15),
                 #    ],
                 # update=nesterov_momentum,
                 # update_momentum=theano.shared(float32(0.9)),
                 update=adagrad,
                 update_learning_rate=theano.shared(float32(0.01)),
                 eval_size=.3,
                 verbose=1,
                 max_epochs=1000)

net0.fit(x_train, y_train)
# prob = net0.predict_proba(x_test)

# df_result = pd.read_csv('../data/sample_submission.csv')
# df_result.loc[:,'TripType_3':'TripType_999'] = prob
# df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
# df_result.to_csv('../data/models/level2_nn_submission.csv', index=False)
