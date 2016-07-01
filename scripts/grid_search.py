import itertools
from multiprocessing import Pool
import numpy as np
#Import libraries:
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier  #GBM algorithm
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 4

from sklearn.cross_validation import train_test_split
import pickle

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb

params_range = {'a': range(1, 5),
          'b': range(6, 9)}

def transform_x(X, x_transformer = None):
    """
    X = [[x, y, a, t]]
    """
    fw = [500., 1000., 4., 3., 2., 10., 10.]
    minute_v = X[:, 3]%60
    hour_v = X[:, 3]//60
    weekday_v = hour_v//24
    month_v = weekday_v//30
    year_v = (weekday_v//365 + 1)*fw[5]
    hour_v = ((hour_v%24 + 1) + minute_v/60.0)*fw[2]
    hour_v_2 = (X[:, 3]%(60*60*24))//(60*60*2)
    hour_v_3 = (X[:, 3]%(60*60*24))//(60*60*3)
    hour_v_4 = (X[:, 3]%(60*60*24))//(60*60*4)
    hour_v_6 = (X[:, 3]%(60*60*24))//(60*60*6)
    hour_v_8 = (X[:, 3]%(60*60*24))//(60*60*8)
    weekday_v = (weekday_v%7 + 1)*fw[3]
    month_v = (month_v%12 +1)*fw[4]
    accuracy_v = np.log10(X[:, 2])*fw[6]
    x_v = X[:, 0]*fw[0]
    y_v = X[:, 1]*fw[1]
    return np.hstack((x_v.reshape(-1, 1),
                     y_v.reshape(-1, 1),
                     accuracy_v.reshape(-1, 1),
                     hour_v.reshape(-1, 1),
                     hour_v_2.reshape(-1, 1),
                     hour_v_3.reshape(-1, 1),
                     hour_v_4.reshape(-1, 1),
                     hour_v_6.reshape(-1, 1),
                     hour_v_8.reshape(-1, 1),
                     weekday_v.reshape(-1, 1),
                     month_v.reshape(-1, 1),
                     year_v.reshape(-1, 1)))

def transform_y(y, y_transformer = None):
    """
    place_ids to encoded array
    """
    y = y.astype(int)
    if y_transformer == None:
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y_transformer = {'encoder': label_encoder}
    new_y = y_transformer['encoder'].transform(y).reshape(-1, 1)
    return (new_y, y_transformer)

def map3eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(1,4)]
    metric = 0.
    for i in range(3):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@3', metric

def load_data(m, n):
    return np.loadtxt('../grid1_400_100_50_10/grid_data_' + str(m)  + '_' + str(n) + '.csv', delimiter = ',')

def get_preds(probs, encoder):
    return encoder.inverse_transform(np.argsort(probs, axis = 1)[:, ::-1][:, :3])

m = 12
n = 40
data = load_data(m, n)
M = pickle.load(open('../grid1_400_100_50_10/cardinality_pickle.pkl', 'rb'))
train, test = train_test_split(data, test_size = 0.02)
mask = np.array(map(lambda x: M[m][n][x] > 20, train[:, 5]))
card_train = train[mask, :]
print card_train.shape, "card_train_shape"
print train.shape, "train_shape"
print test.shape, "test_shape"

X = transform_x(card_train[:, (1, 2, 3, 4)])
y_orig, enc = transform_y(data[:, 5])
y, enc = transform_y(card_train[:, 5], enc)
test_X = transform_x(test[:, (1, 2, 3, 4)])
test_y, enc = transform_y(test[:, 5], enc)
print X.shape, "X shape"
print y.shape, "y shape"
print test_X.shape, "test_X shape"
print test_y.shape, "test_y shape"
print len(enc['encoder'].classes_)

dtrain = xgb.DMatrix(X, label=np.ravel(y))
dtest = xgb.DMatrix(test_X, label=np.ravel(test_y))


tup = lambda t: list(itertools.izip(itertools.repeat(t[0]), t[1]))

def get_list_of_params(params_range):
    pr = list(map(tup, params_range.items()))
    pro = map(dict, list(itertools.product(*pr)))
    return pro


def grid_search_xgb(params_range_dict):
    grid_params_list = get_list_of_params(params_range_dict)
    p = Pool(4)
    maps = p.map(get_map_of_xgb, grid_params_list)
    sorted_maps = sorted(maps, cmp = lambda x, y: cmp(x['map'], y['map']), reverse = True)
    print "top map results", sorted_maps[:3]
    return sorted_maps

def get_map_of_xgb(grid_param):
    orig_params = {'num_class': len(enc['encoder'].classes_),
                     'silent': 0,
                     'nthread': 8,
                     'eta': 0.1,
                     'objective': 'multi:softprob',
                     'max_depth': 8,
                     'min_child_weight': 5,
                     'gamma': 0.32,
                     'subsample': 0.9,
                     'colsample_bytree': 0.7,
                     'scale_pos_weight': 1
                  }
    orig_params.update(grid_param)
    # print orig_params, grid_param
    temp_cv = xgb.cv(orig_params, dtrain, num_boost_round = 100,
             early_stopping_rounds = 20, feval = map3eval, maximize = True)
    temp_map = temp_cv['test-MAP@3-mean'][temp_cv.shape[0]-1]
    grid_param['map'] = temp_map
    print "cv results", grid_param
    return grid_param


# Can't freaking pickle DMatrix
# class GridCVStateLoader(object):
#     def __init__(self, dtrain, enc):
#         self.dtrain = dtrain
#         self.enc = enc
#     def __call__(self, grid_param):
#         return get_map_of_xgb(grid_param, self.dtrain, self.enc)
