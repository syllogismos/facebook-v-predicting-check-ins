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

import traceback
import grid_generation as grid


import logging
logging.basicConfig(filename='param_search.log',level=logging.DEBUG)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("X", type=int)
parser.add_argument("Y", type=int)
parser.add_argument("xd", type=int)
parser.add_argument("yd", type=int)
parser.add_argument("m", type=int)
parser.add_argument("n", type=int)
parser.add_argument("th", type=int)
args = parser.parse_args()

m = args.m #12
n = args.n #40
th = args.th #3

# g = grid.Grid(200, 50, 20, 5, pref = 'grid')
g = grid.Grid(args.X, args.Y, args.xd, args.yd, pref = 'grid')
g.generateCardinalityMatrix()


orig_params = {
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


def transform_x(X, x_transformer = None):
    """
    X = [[x, y, a, t]]
    """
    minute_v = X[:, 3]%60
    hour_v = X[:, 3]//60
    weekday_v = hour_v//24
    month_v = weekday_v//30
    year_v = (weekday_v//365 + 1)
    hour_v = ((hour_v%24 + 1) + minute_v/60.0)
    hour_v_2 = (X[:, 3]%(60*60*24))//(60*60*2)
    hour_v_3 = (X[:, 3]%(60*60*24))//(60*60*3)
    hour_v_4 = (X[:, 3]%(60*60*24))//(60*60*4)
    hour_v_6 = (X[:, 3]%(60*60*24))//(60*60*6)
    hour_v_8 = (X[:, 3]%(60*60*24))//(60*60*8)
    weekday_v = (weekday_v%7 + 1)
    month_v = (month_v%12 +1)
    accuracy_v = np.log10(X[:, 2])
    x_v = X[:, 0]
    y_v = X[:, 1]
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
    f = g.getGridFile(m, n)
    return np.loadtxt(f, delimiter = ',')

def get_preds(probs, encoder):
    return encoder.inverse_transform(np.argsort(probs, axis = 1)[:, ::-1][:, :3])


def get_dtrain_enc(m, n):
    data = load_data(m, n)
    M = g.M

    mask = np.array(map(lambda x: M[m][n][x] > th, data[:, 5]))
    train = data[mask, :]
    print data.shape, "data_shape"


    X = transform_x(train[:, (1, 2, 3, 4)])
    y, enc = transform_y(train[:, 5])
    print X.shape, "X shape"
    print y.shape, "y shape"
    print len(enc['encoder'].classes_), "no of classes"
    logging.debug("no of classes: %s" %(len(enc['encoder'].classes_)))

    dtrain = xgb.DMatrix(X, label=np.ravel(y))
    return (dtrain, enc)







tup = lambda t: list(itertools.izip(itertools.repeat(t[0]), t[1]))

def get_list_of_params(params_range):
    pr = list(map(tup, params_range.items()))
    pro = map(dict, list(itertools.product(*pr)))
    return pro


def grid_search_xgb(params_range_dict):
    grid_params_list = get_list_of_params(params_range_dict)
    p = Pool(8)
    maps = p.map(get_map_of_xgb, grid_params_list)
    p.close()
    p.join()
    sorted_maps = sorted(maps, cmp = lambda x, y: cmp(x['map'], y['map']), reverse = True)
    logging.debug("top map results")
    logging.debug(sorted_maps[:3])
    return sorted_maps

def get_map_of_xgb(grid_param):
    cv_params = dict(orig_params)
    num_class = {'num_class': len(enc['encoder'].classes_)}
    cv_params.update(num_class)
    cv_params.update(grid_param)
    # print orig_params, grid_param
    temp_cv = xgb.cv(cv_params, dtrain, num_boost_round = 100,
             early_stopping_rounds = 20, feval = map3eval, maximize = True)
    temp_map = temp_cv['test-MAP@3-mean'][temp_cv.shape[0]-1]
    grid_param['map'] = temp_map
    # print "cv results", grid_param
    return grid_param

#m, n = (12, 50) (37, 50) (12, 150) (37, 150)
#dtrain, enc = get_dtrain_enc(12, 50)
param_range1 = {
    'max_depth': range(2, 10, 1),
    'min_child_weight': range(1, 7, 1)
}

param_range2 = {
    'gamma': [i/10.0 for i in range(0, 6)],
    'colsample_bylevel': [i/10.0 for i in range(5, 10)]
}

param_range3 = {
    'max_delta_step': range(3, 15)
}

param_range4 = {
    'subsample': [i/100.0 for i in range(60, 100, 5)],
    'colsample_bytree': [i/100.0 for i in range(60, 100, 5)]
}



if __name__ == '__main__':
    orig_params = {
                'nthread': 4,
                'eta': 0.1,
                'objective': 'multi:softprob',
                'max_depth': 8,
                'min_child_weight': 5,
                'gamma': 0.32,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'scale_pos_weight': 1,
                'silent': 1
    }
    logging.debug("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
    dtrain, enc = get_dtrain_enc(m, n)
    for param_range in [param_range1, param_range2, param_range3,\
        param_range4]:
        result = None
        try:
            result = grid_search_xgb(param_range)
        except Exception, e:
            print e
            print traceback.format_exc()
        if result != None:
            temp_param = result[0]
            del(temp_param['map'])
            orig_params.update(temp_param)
            logging.debug("best params so far")
            logging.debug(orig_params)

    logging.debug("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    logging.debug("best params for the grid %s, %s, %s, %s" %(g.X, g.Y, g.xd, g.yd))
    logging.debug("grid no m, n %s, %s" %(m, n))
    logging.debug("threshold: %s" %(th))
    logging.debug(orig_params)
    logging.debug("###################################################")
