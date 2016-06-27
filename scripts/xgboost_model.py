
import time
import xgboost as xgb
from grid_generation import Grid, generate_grid_wise_cardinality_and_training_files
from grid_generation import get_top_3_places_of_dict, get_grids
from helpers import days, hours, quarter_days
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from helpers import apk

from base_scikit_learn_model import SklearnModel

import numpy as np
import os
import pickle

from multiprocessing import Pool

grid = Grid(X = 400, Y = 100, xd = 50, yd = 10, pref = 'grid1', train_file = '../main_train_0.02_5.csv')

def map3eval(preds, dtrain):
    actual = dtrain.get_label()
    predicted = preds.argsort(axis=1)[:,-np.arange(1,4)]
    metric = 0.
    for i in range(3):
        metric += np.sum(actual==predicted[:,i])/(i+1)
    metric /= actual.shape[0]
    return 'MAP@3', metric

class XGB_Model(SklearnModel):

    def transform_x(X, x_transformer = None):
        """
        X = [[x, y, a, t]]
        """
        fw = [1., 1., 1., 1., 1., 1., 1.]
        minute_v = X[:, 3]%60
        hour_v = X[:, 3]//60
        weekday_v = hour_v//24
        month_v = weekday_v//30
        year_v = (weekday_v//365 + 1)*fw[5]
        hour_v = ((hour_v%24 + 1) + minute_v/60.0)*fw[2]
        weekday_v = (weekday_v%7 + 1)*fw[3]
        month_v = (month_v%12 +1)*fw[4]
        accuracy_v = np.log10(X[:, 2])*fw[6]
        x_v = X[:, 0]*fw[0]
        y_v = X[:, 1]*fw[1]
        X_new = np.hstack((x_v.reshape(-1, 1),\
                         y_v.reshape(-1, 1),\
                         accuracy_v.reshape(-1, 1),\
                         hour_v.reshape(-1, 1),\
                         weekday_v.reshape(-1, 1),\
                         month_v.reshape(-1, 1),\
                         year_v.reshape(-1, 1)))
        return (X_new, x_transformer)


    def custom_classifier(self, X, Y, y_transformer):
        param = {}
        # use softmax multi-class classification
        param['objective'] = 'multi:softprob'
        # scale weight of positive examples
        param['eta'] = 0.1
        param['max_depth'] = 6
        param['silent'] = 1
        param['nthread'] = 4
        param['num_class'] = len(y_transformer['encoder'].classes_)
        param['eval_metric'] = ['merror', 'mlogloss']
        num_round = 25
        dtrain = xgb.DMatrix(X, label=np.ravel(Y))
        return xgb.train(param, dtrain, num_round, feval = map3eval)

    # def train_row(self, i):
    #     return map(lambda x: self.train_grid(i, x), range(self.grid.max_n + 1))

    # def train(self):
    #     pool = Pool(processes = 4)
    #     m = self.grid.max_m + 1
    #     n = self.grid.max_n + 1
    #     result = pool.apply_async(self.train_row, range(m))
    #     result.get()
    #     pass

    def train_grid(self, m, n):
        """
        Helper function for train function that takes the grid cell to be trained
        """
        # print "Training %s, %s grid" %(m, n)
        init_time = time.time()
        data = np.loadtxt(self.grid.getGridFile(m, n), dtype = float, delimiter = ',')
        if len(data) == 0 or len(data.shape) == 1:
            # if the data contains only one row then also make model None
            # if the grid contains barely zero data, make model None
            self.model[m][n]['model'] = None
            return
        mask = np.array(map(lambda x: self.grid.M[m][n][x] > self.threshold, data[:, 5]))
        masked_data = data[mask, :]
        if len(masked_data) < 10:
            self.model[m][n]['model'] = None
            return
        X, x_transformer = self.transform_x(masked_data[:, (1, 2, 3, 4)])
        Y, y_transformer = self.transform_y(masked_data[:, 5])

        self.model[m][n]['x_transformer'] = x_transformer
        self.model[m][n]['y_transformer'] = y_transformer

        if len(Y) == 0:
            # if masked data is of length zero then also make model None
            self.model[m][n]['model'] = None
        else:
            self.model[m][n]['model'] = self.custom_classifier(X, Y, y_transformer)
        # return True

        # print "Time taken to train grid %s, %s is: %s" %(m, n, time.time() - init_time)
        # print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

    def predict_grid(self, grid_data, m, n):
        """
        grid_data is test/cv data from that particular grid
        return row_id, and top 3 predictions
        """
        # print "predicting grid %s, %s" %(m, n)
        grid_data = np.array(grid_data)
        if self.model[m][n]['model'] == None:
            top_3_placeids = np.array([[5348440074, 9988088517, 4048573921]]*len(grid_data))
        else:
            temp_x = self.transform_x(grid_data[:, (1, 2, 3, 4)], self.model[m][n]['x_transformer'])[0]
            dtest = xgb.DMatrix(temp_x)
            prediction_probs = self.model[m][n]['model'].predict(dtest)
            top_3_placeids = self.model[m][n]['y_transformer']['encoder'].inverse_transform(np.argsort(prediction_probs, axis=1)[:,::-1][:,:3])

            # temporary hack when the no of predictions for a row is less than 3
            x, y = top_3_placeids.shape
            if y < 3:
                temp_array = np.array([[5348440074]*(3-y)]*len(top_3_placeids))
                top_3_placeids = np.hstack((top_3_placeids, temp_array))

        return np.hstack((grid_data[:, 0].reshape(-1, 1), top_3_placeids))

    def train_first_grid_and_predict(self, m = 0, n = 0):
        """
        """
        data = np.loadtxt(self.grid.getGridFile(m, n), dtype = float, delimiter = ',')
        train, test = train_test_split(data, test_size = 0.09)
        ty, y_transformer = self.transform_y(data[:, 5])
        mask = np.array(map(lambda x: self.grid.M[m][n][x] > self.threshold, train[:, 5]))
        masked_train = train[mask, :]
        X, x_transformer = self.transform_x(masked_train[:, (1, 2, 3, 4)])
        Y, y_transformer = self.transform_y(masked_train[:, 5], y_transformer)

        test_X = self.transform_x(test[:, (1, 2, 3, 4)], x_transformer)[0]

        trained_clf = self.custom_classifier(X, Y, y_transformer)

        print "Predicting the probablity of train set"
        cv_mean_precision_train = self.get_cv(trained_clf, masked_train, y_transformer)
        print cv_mean_precision_train, "training data prediction precision"

        print "Predicting the probability of test set"
        cv_mean_precision_test = self.get_cv(trained_clf, test, y_transformer)
        print cv_mean_precision_test, "test data prediction precision"

    def get_cv(self, clf, data, y_transformer):
        """
        """
        X = self.transform_x(data[:, (1, 2, 3, 4)])[0]
        DMX = xgb.DMatrix(X)
        prediction_probs = clf.predict(DMX)
        top_3 = y_transformer['encoder'].inverse_transform(np.argsort(prediction_probs, axis=1)[:,::-1][:,:3])
        actual = data[:, -1].astype(int).reshape(-1, 1)
        preds = np.hstack((top_3, actual))
        print preds[:5]
        apk_list = map(lambda row: apk(row[-1:], row[: -1]), preds)
        return np.mean(apk_list)


