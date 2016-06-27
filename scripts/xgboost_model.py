
import time
import xgboost as xgb
from grid_generation import Grid, generate_grid_wise_cardinality_and_training_files
from grid_generation import get_top_3_places_of_dict, get_grids
from helpers import days, hours, quarter_days
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from helpers import apk, zip_file_and_upload_to_s3

from base_scikit_learn_model import SklearnModel, get_grids_of_a_point

import numpy as np
import os
import pickle
import pdb

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

class StateLoader(object):
    def __init__(self, state):
        self.state = state
    def __call__(self, row):
        return train_row(row, self.state)
"""
State:
    grid
    cv_grid
    test_grid
state = {
    'grid': Grid,
    'cv_grid': mxn matrix
    'test_grid': mxn matrix
    'threshold': 5
"""
def train_row(i, state):
    print "processing row %s" %(i)
    init_row_time = time.time()
    test_preds = []
    cv_preds = []
    for n in range(state['grid'].max_n + 1):
        if n % 10 == 0:
            print "processing column %s of row %s" %(n, i)
        clf, x_transformer, y_transformer = train_single_grid_cell(i, n, state)
        if len(state['test_grid'][i][n]) > 0:
            test_preds.append(predict_single_grid_cell(state['test_grid'][i][n], \
                clf, x_transformer, y_transformer))
        if len(state['cv_grid'][i][n]) > 0:
            cv_preds.append(predict_single_grid_cell(state['cv_grid'][i][n], \
                clf, x_transformer, y_transformer))

    if len(test_preds) > 0:
        test_row = np.vstack(test_preds)
    else:
        test_row = None
    if len(cv_preds) > 0:
        cv_row = np.vstack(cv_preds)
    else:
        cv_row = None

    print "time taken for row %s is %s" %(i, time.time() - init_row_time())
    return (test_row, cv_row)

def train_single_grid_cell(m, n, state):
    data = np.loadtxt(state['grid'].getGridFile(m, n), dtype = float, delimiter = ',')
    if len(data) == 0 or len(data.shape) == 1:
        return None, None, None
    mask = np.array(map(lambda x: state['grid'].M[m][n][x] > state['threshold'], data[:, 5]))
    masked_data = data[mask, :]
    if len(masked_data) < 10:
        return None, None, None
    X, x_transformer = trans_x(masked_data[:, (1, 2, 3, 4)])
    Y, y_transformer = trans_y(masked_data[:, 5])

    if len(Y) == 0:
        return None, None, None
    else:
        return classifier(X, Y, y_transformer), x_transformer, y_transformer
    pass

def predict_single_grid_cell(X, clf, x_transformer, y_transformer):
    data = np.array(X)
    if clf == None:
        top_3_placeids = np.array([[5348440074, 9988088517, 4048573921]]*len(data))
    else:
        temp_x = trans_x(data[:, (1, 2, 3, 4)], x_transformer)[0]
        dtest = xgb.DMatrix(temp_x)
        prediction_probs = clf.predict(dtest)
        top_3_placeids = y_transformer['encoder'].inverse_transform(np.argsort(prediction_probs, axis = 1)[:, ::-1][:, :3])
        x, y = top_3_placeids.shape
        if y < 3:
            temp_array = np.array([[5348440074]*(3-y)]*len(top_3_placeids))
            top_3_placeids = np.hstack((top_3_placeids, temp_array))
    return np.hstack((data[:, 0].reshape(-1, 1), top_3_placeids))

def trans_x(X, x_transformer = None):
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

def trans_y(y, y_transformer = None):
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

def classifier(X, Y, y_transformer):
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

class XGB_Model(SklearnModel):

    def transform_x(self, X, x_transformer = None):
        """
        X = [[x, y, a, t]]
        """
        return trans_x(X, x_transformer)
        # fw = [1., 1., 1., 1., 1., 1., 1.]
        # minute_v = X[:, 3]%60
        # hour_v = X[:, 3]//60
        # weekday_v = hour_v//24
        # month_v = weekday_v//30
        # year_v = (weekday_v//365 + 1)*fw[5]
        # hour_v = ((hour_v%24 + 1) + minute_v/60.0)*fw[2]
        # weekday_v = (weekday_v%7 + 1)*fw[3]
        # month_v = (month_v%12 +1)*fw[4]
        # accuracy_v = np.log10(X[:, 2])*fw[6]
        # x_v = X[:, 0]*fw[0]
        # y_v = X[:, 1]*fw[1]
        # X_new = np.hstack((x_v.reshape(-1, 1),\
        #                  y_v.reshape(-1, 1),\
        #                  accuracy_v.reshape(-1, 1),\
        #                  hour_v.reshape(-1, 1),\
        #                  weekday_v.reshape(-1, 1),\
        #                  month_v.reshape(-1, 1),\
        #                  year_v.reshape(-1, 1)))
        # return (X_new, x_transformer)


    def custom_classifier(self, X, Y, y_transformer):
        classifier(X, Y, y_transformer)
        # param = {}
        # # use softmax multi-class classification
        # param['objective'] = 'multi:softprob'
        # # scale weight of positive examples
        # param['eta'] = 0.1
        # param['max_depth'] = 6
        # param['silent'] = 1
        # param['nthread'] = 4
        # param['num_class'] = len(y_transformer['encoder'].classes_)
        # param['eval_metric'] = ['merror', 'mlogloss']
        # num_round = 25
        # dtrain = xgb.DMatrix(X, label=np.ravel(Y))
        # return xgb.train(param, dtrain, num_round, feval = map3eval)

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

    def train_and_predict_parallel(self, submission_file, upload_to_s3 = False):
        init_time = time.time()
        cv_data = np.loadtxt(self.cross_validation_file, dtype = float, delimiter = ',')
        test_data = np.loadtxt(self.test_file, dtype = float, delimiter = ',')
        test_grid_wise_data = [[[] for n in range(self.grid.max_n + 1)]\
            for m in range(self.grid.max_m + 1)]
        cv_grid_wise_data = [[[] for n in range(self.grid.max_n + 1)]\
            for m in range(self.grid.max_m + 1)]

        print "converting test data to grid wise"
        for i in range(len(test_data)):
            m, n = get_grids_of_a_point((test_data[i][1], test_data[i][2]), self.grid)[0]
            test_grid_wise_data[m][n].append(test_data[i])

        print "converting cv data to grid wise"
        for i in range(len(cv_data)):
            m, n = get_grids_of_a_point((cv_data[i][1], cv_data[i][2]), self.grid)[0]
            cv_grid_wise_data[m][n].append(cv_data[i])

        state = {}
        state['grid'] = self.grid
        state['cv_grid'] = cv_grid_wise_data
        state['test_grid'] = test_grid_wise_data
        state['threshold'] = self.threshold

        p = Pool(12)
        row_results = p.map(StateLoader(state), range(self.grid.max_m + 1))
        print "Training time of parallel processing %s" %(time.time() - init_time)
        # row_results = map(StateLoader(state), range(self.grid.max_m - 1, self.grid.max_m + 1))
        # pdb.set_trace()
        test_rows = map(lambda x: x[0], row_results)
        cv_rows = map(lambda x: x[1], row_results)

        test_rows = filter(lambda x: x != None, test_rows)
        cv_rows = filter(lambda x: x != None, cv_rows)

        # pdb.set_trace()
        test_preds = np.vstack(test_rows).astype(int)
        cv_preds = np.vstack(cv_rows).astype(int)

        sorted_test = test_preds[test_preds[:, 0].argsort()]
        sorted_cv = cv_preds[cv_preds[:, 0].argsort()]

        actual_cv = cv_data[:, -1].astype(int).reshape(-1, 1)
        cv_a_p = np.hstack((sorted_cv, actual_cv))
        apk_list = map(lambda row: apk(row[-1:], row[1:-1]), cv_a_p)
        self.cv_mean_precision = np.mean(apk_list)
        print "mean precision of cross validation set", str(self.cv_mean_precision)

        sorted_test = sorted_test.astype(str)
        submission = open(submission_file, 'wb')
        submission.write('row_id,place_id\n')
        for i in range(len(sorted_test)):
            row = sorted_test[i]
            row_id = row[0]
            row_prediction_string = ' '.join(row[1:])
            submission.write(row_id + ',' + row_prediction_string + '\n')
            if i % 1000000 == 0:
                print "generating %s row of test data" %(i)
        submission.close()
        if upload_to_s3:
            zip_file_and_upload_to_s3(submission_file)

    def train_and_predict(self, submission_file, upload_to_s3 = False):
        cv_data = np.loadtxt(self.cross_validation_file, dtype = float, delimiter = ',')
        test_data = np.loadtxt(self.test_file, dtype = float, delimiter = ',')
        max_m = self.grid.max_m
        max_n = self.grid.max_n
        test_grid_wise_data = [[[] for n in range(max_n + 1)]\
            for m in range(max_m + 1)]
        cv_grid_wise_data = [[[] for n in range(max_n + 1)]\
            for m in range(max_m + 1)]

        print "converting test data to grid wise"
        for i in range(len(test_data)):
            m, n = get_grids_of_a_point((test_data[i][1], test_data[i][2]), self.grid)[0]
            test_grid_wise_data[m][n].append(test_data[i])

        print "converting cv data to grid wise"
        for i in range(len(cv_data)):
            m, n = get_grids_of_a_point((cv_data[i][1], cv_data[i][2]), self.grid)[0]
            cv_grid_wise_data[m][n].append(cv_data[i])

        test_preds = []
        cv_preds = []

        for m in range(max_m + 1):
            print "row %s training and predicting" %(m)
            for n in range(max_n + 1):
                self.train_grid(m, n)
                if len(test_grid_wise_data[m][n]) > 0:
                    test_preds.append(self.predict_grid(np.array(test_grid_wise_data[m][n]), m, n))
                if len(cv_grid_wise_data[m][n]) > 0:
                    cv_preds.append(self.predict_grid(np.array(cv_grid_wise_data[m][n]), m, n))
                self.model[m][n]['model'] = None
        # pdb.set_trace()
        test_preds = np.vstack(tuple(test_preds)).astype(int)
        cv_preds = np.vstack(tuple(cv_preds)).astype(int)

        sorted_test = test_preds[test_preds[:, 0].argsort()]
        sorted_cv = cv_preds[cv_preds[:, 0].argsort()]

        actual_cv = cv_data[:, -1].astype(int).reshape(-1, 1)
        cv_a_p = np.hstack((sorted_cv, actual_cv))
        apk_list = map(lambda row: apk(row[-1:], row[1:-1]), cv_a_p)
        self.cv_mean_precision = np.mean(apk_list)
        print "mean precision of cross validation set", str(self.cv_mean_precision)

        sorted_test = sorted_test.astype(str)
        submission = open(submission_file, 'wb')
        submission.write('row_id,place_id\n')
        for i in range(len(sorted_test)):
            row = sorted_test[i]
            row_id = row[0]
            row_prediction_string = ' '.join(row[1:])
            submission.write(row_id + ',' + row_prediction_string + '\n')
            if i % 1000000 == 0:
                print "generating %s row of test data" %(i)
        submission.close()
        if upload_to_s3:
            zip_file_and_upload_to_s3(submission_file)


