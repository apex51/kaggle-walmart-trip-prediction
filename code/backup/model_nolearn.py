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

# load from file
with open('../data/pack.pkl', 'rb') as f:
    x_train, y_train, x_test = pickle.load(f)

with open('../data/feat_list.pkl', 'rb') as f:
    feat_list = pickle.load(f)

x_train = x_train[:, feat_list]
x_test = x_test[:, feat_list]

x_train = x_train.toarray().astype(np.float32)
x_test = x_test.toarray().astype(np.float32)
y_train = y_train.astype(np.int32)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

num_features = x_train.shape[-1]


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
                 dropoutf_p=0.02,
                 dense0_num_units=600,
                 dropout_p=0.75,
                 # dense1_num_units=600,
                 # dropout2_p=0.05,
                 output_num_units=38,
                 output_nonlinearity=softmax,
                 # on_epoch_finished=[
                 #    EarlyStopping(patience=20),
                 #    ],
                 # update=nesterov_momentum,
                 # update_momentum=theano.shared(float32(0.9)),
                 update=adagrad,
                 update_learning_rate=theano.shared(float32(0.01)),
                 eval_size=.0,
                 verbose=1,
                 max_epochs=35)

net0.fit(x_train, y_train)
pred = net0.predict_proba(x_test)

df_result = pd.read_csv('../data/sample_submission.csv')
df_result.loc[:,'TripType_3':'TripType_999'] = pred
df_result[['VisitNumber']] = df_result[['VisitNumber']].astype(int)
df_result.to_csv('../submission.csv', index=False)

