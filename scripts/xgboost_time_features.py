
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
    'params_dict': xgb params for each grid
    'folder': for giggles, i mean to save files
"""
def train_row(i, state):
    print "processing row %s" %(i)
    init_row_time = time.time()
    test_preds = []
    cv_preds = []
    train_preds = []
    for n in range(state['grid'].max_n + 1):
    # for n in range(1):
        if n % 10 == 0:
            print "processing column %s of row %s" %(n, i)
        clf, x_transformer, y_transformer, top_t, top_t_train_places = train_single_grid_cell(i, n, state)
        if top_t_train_places != None:
            train_preds.append(top_t_train_places)
        if len(state['test_grid'][i][n]) > 0:
            test_preds.append(predict_single_grid_cell(state['test_grid'][i][n], \
                clf, x_transformer, y_transformer, top_t))
        if len(state['cv_grid'][i][n]) > 0:
            cv_preds.append(predict_single_grid_cell(state['cv_grid'][i][n], \
                clf, x_transformer, y_transformer, top_t))
        del(clf)

    if len(test_preds) > 0:
        test_row = np.vstack(test_preds)
    else:
        test_row = None
    if len(cv_preds) > 0:
        cv_row = np.vstack(cv_preds)
    else:
        cv_row = None
    if len(train_preds) > 0:
        train_row_preds = np.vstack(train_preds)
    else:
        train_row_preds = None
    print "time taken for row %s is %s" %(i, time.time() - init_row_time)
    return (test_row, cv_row, train_row_preds)

def train_single_grid_cell(m, n, state):
    # print m, n
    t = 10
    folder = state['folder']
    data = np.loadtxt(state['grid'].getGridFile(m, n), dtype = float, delimiter = ',')
    top_t = sorted(state['grid'].M[m][n].items(), cmp = lambda x, y: cmp(x[1], y[1]), reverse = True)[:t]
    top_t = map(lambda x: x[0], top_t)
    y = len(top_t)
    if y < t:
        top_t += [123]*(t-y) #[5348440074]*(t-y)
    if len(data) == 0:
        return None, None, None, top_t, None
    if len(data.shape) == 1:
        top_t_train_preds = np.hstack((data[:1], top_t))
        top_t_train_preds = top_t_train_preds.astype(int)
        file_name = folder + '_'.join(['top_t_preds', str(m), str(n)]) + '.csv'
        np.savetxt(file_name, top_t_train_preds, fmt = '%s', delimiter = ',')
        return None, None, None, top_t, top_t_train_preds
    mask = np.array(map(lambda x: state['grid'].M[m][n][x] > state['threshold'], data[:, 5]))
    masked_data = data[mask, :]
    if len(masked_data) < 10:
        top_t_train_preds = np.hstack((data[:, 0].reshape(-1, 1), [top_t]*len(data)))
        top_t_train_preds = top_t_train_preds.astype(int)
        file_name = folder + '_'.join(['top_t_preds', str(m), str(n)]) + '.csv'
        np.savetxt(file_name, top_t_train_preds, fmt = '%s', delimiter = ',')
        return None, None, None, top_t, top_t_train_preds
    X, x_transformer = trans_x(masked_data[:, (1, 2, 3, 4)])
    Y, y_transformer = trans_y(masked_data[:, 5])

    if len(Y) == 0:
        top_t_train_preds = np.hstack((data[:, 0].reshape(-1, 1), [top_t]*len(data)))
        top_t_train_preds = top_t_train_preds.astype(int)
        file_name = folder + '_'.join(['top_t_preds', str(m), str(n)]) + '.csv'
        np.savetxt(file_name, top_t_train_preds, fmt = '%s', delimiter = ',')
        return None, None, None, top_t, top_t_train_preds
    else:
        params = dict(state['params_dict'][m][n])
        params['num_class'] = len(y_transformer['encoder'].classes_)
        bst = classifier(X, Y, params)
        X_orig, x_transformer = trans_x(data[:, (1, 2, 3, 4)], x_transformer)
        dtrain_orig = xgb.DMatrix(X_orig)
        train_preds_proba = bst.predict(dtrain_orig)
        if len(train_preds_proba.shape) == 1:
            train_preds_proba = train_preds_proba.reshape(-1, 1)
        top_t_train_preds = y_transformer['encoder'].inverse_transform(np.argsort(train_preds_proba, axis = 1)[:, ::-1][:, :t])
        x, y = top_t_train_preds.shape
        if y < t:
            temp_array = [[123]*(t-y)]*x
            top_t_train_preds= np.hstack((top_t_train_preds, temp_array))
        top_t_train_preds = np.hstack((data[:, 0].reshape(-1, 1), top_t_train_preds))
        top_t_train_preds = top_t_train_preds.astype(int)
        file_name = folder + '_'.join(['top_t_preds', str(m), str(n)]) + '.csv'
        np.savetxt(file_name, top_t_train_preds, fmt = '%s', delimiter = ',')
        return bst, x_transformer, y_transformer, top_t, top_t_train_preds
    pass

def predict_single_grid_cell(X, clf, x_transformer, y_transformer, top_t):
    t = 10
    data = np.array(X)
    if clf == None:
        top_t_placeids = np.array([top_t]*len(data))
    else:
        temp_x = trans_x(data[:, (1, 2, 3, 4)], x_transformer)[0]
        dtest = xgb.DMatrix(temp_x)
        prediction_probs = clf.predict(dtest)
        if len(prediction_probs.shape) == 1:
            prediction_probs = prediction_probs.reshape(-1, 1)
        top_t_placeids = y_transformer['encoder'].inverse_transform(np.argsort(prediction_probs, axis = 1)[:, ::-1][:, :t])
        x, y = top_t_placeids.shape
        if y < t:
            temp_array = np.array([top_t[:(t-y)]]*len(top_t_placeids))
            top_t_placeids = np.hstack((top_t_placeids, temp_array))
    return np.hstack((data[:, 0].reshape(-1, 1), top_t_placeids))

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
    hour_v_2 = (X[:, 3]%(60*60*24))//(60*60*2)
    hour_v_3 = (X[:, 3]%(60*60*24))//(60*60*3)
    hour_v_4 = (X[:, 3]%(60*60*24))//(60*60*4)
    hour_v_6 = (X[:, 3]%(60*60*24))//(60*60*6)
    hour_v_8 = (X[:, 3]%(60*60*24))//(60*60*8)
    weekday_v = (weekday_v%7 + 1)*fw[3]
    month_v = (month_v%12 +1)*fw[4]
    accuracy_v = np.log10(X[:, 2])*fw[6]

    minute_5 = 2*np.pi*((X[:, 3]//5)%288)/288
    min_sin = (np.sin(minute_5) + 1).round(4)
    min_cos = (np.cos(minute_5) + 1).round(4)

    day_of_year = 2*np.pi*((X[:, 3]//1440)%365)/365
    day_of_year_sin = (np.sin(day_of_year) + 1).round(4)
    day_of_year_cos = (np.cos(day_of_year) + 1).round(4)

    weekday = 2*np.pi*((X[:, 3]//1440)%7)/7
    weekday_sin = (np.sin(weekday) + 1).round(4)
    weekday_cos = (np.cos(weekday) + 1).round(4)

    x_v = X[:, 0]*fw[0]
    y_v = X[:, 1]*fw[1]
    X_new = np.hstack((x_v.reshape(-1, 1),\
                     y_v.reshape(-1, 1),\
                     accuracy_v.reshape(-1, 1),\
                     min_sin.reshape(-1, 1),\
                     min_cos.reshape(-1, 1),\
                     day_of_year_sin.reshape(-1, 1),\
                     day_of_year_cos.reshape(-1, 1),\
                     weekday_sin.reshape(-1, 1),\
                     weekday_cos.reshape(-1, 1),\
                     hour_v.reshape(-1, 1),\
                     hour_v_2.reshape(-1, 1),\
                     hour_v_3.reshape(-1, 1),\
                     hour_v_4.reshape(-1, 1),\
                     hour_v_6.reshape(-1, 1),\
                     hour_v_8.reshape(-1, 1),\
                     weekday_v.reshape(-1, 1),\
                     month_v.reshape(-1, 1),\
                     year_v.reshape(-1, 1)))
    return (X_new, x_transformer)

#     minute = 2*np.pi*((df["time"]//5)%288)/288
#     df['minute_sin'] = (np.sin(minute)+1).round(4)
#     df['minute_cos'] = (np.cos(minute)+1).round(4)
#     del minute
#     day = 2*np.pi*((df['time']//1440)%365)/365
#     df['day_of_year_sin'] = (np.sin(day)+1).round(4)
#     df['day_of_year_cos'] = (np.cos(day)+1).round(4)
#     del day
#     weekday = 2*np.pi*((df['time']//1440)%7)/7
#     df['weekday_sin'] = (np.sin(weekday)+1).round(4)
#     df['weekday_cos'] = (np.cos(weekday)+1).round(4)
#     del weekday
#     df['year'] = (df['time']//525600).astype(float)
#     df.drop(['time'], axis=1, inplace=True)
#     df['accuracy'] = np.log10(df['accuracy']).astype(float)
#     return df

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

def classifier(X, Y, params):
    num_round = 100
    dtrain = xgb.DMatrix(X, label=np.ravel(Y))
    bst = xgb.train(params, dtrain, num_round, feval = map3eval)
    return bst

class XGB_Model(SklearnModel):

    def transform_x(self, X, x_transformer = None):
        """
        X = [[x, y, a, t]]
        """
        return trans_x(X, x_transformer)

    def custom_classifier(self, X, Y, params):
        classifier(X, Y, params)

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
            params = {
                'objective': 'multi:softprob',
                'eta': 0.1,
                'max_depth': 13,
                'min_child_weight': 5,
                'gamma': 0.3,
                'subsample': 0.9,
                'colsample_bytree': 0.7,
                'scale_pos_weight': 1,
                'nthread': 4,
                'silent': 1,
                'num_class': len(y_transformer['encoder'].classes_)
            }
            self.model[m][n]['model'] = self.custom_classifier(X, Y, params)
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

        params = {
            'objective': 'multi:softprob',
            'eta': 0.1,
            'max_depth': 13,
            'min_child_weight': 5,
            'gamma': 0.3,
            'subsample': 0.9,
            'colsample_bytree': 0.7,
            'scale_pos_weight': 1,
            'nthread': 4,
            'silent': 1,
            'num_class': len(y_transformer['encoder'].classes_)
        }

        trained_clf = self.custom_classifier(X, Y, params)

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

        default_xgb_params = {
            'objective': 'multi:softprob',
            'eta': 0.1,
            'max_depth': 3,
            'min_child_weight': 6,
            'gamma': 0.1,
            'subsample': 0.9,
            'colsample_bytree': 0.6,
            'scale_pos_weight': 1,
            'nthread': 4,
            'silent': 1,
            'max_delta_step': 7
        }

        paramsFile = self.grid.getParamsFile(5, 123340)
        if paramsFile == None:
            print "params file doesn't exist.. so loading default params"
            state['params_dict'] = [[default_xgb_params for n in range(self.grid.max_n + 1)]\
                                        for m in range(self.grid.max_m + 1)]
        else:
            state['params_dict'] = pickle.load(open(paramsFile, 'rb'))

        submission_name = os.path.basename(submission_file)[:-4]

        folder = self.grid.getFolder()[:-1] + '_' + submission_name + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        state['folder'] = folder

        p = Pool(8)
        row_results = p.map(StateLoader(state), range(self.grid.max_m + 1))
        p.close()
        p.join()
        del(test_grid_wise_data)
        del(cv_grid_wise_data)
        print "Training time of parallel processing %s" %(time.time() - init_time)
        # row_results = map(StateLoader(state), range(1))
        # pdb.set_trace()
        test_rows = map(lambda x: x[0], row_results)
        cv_rows = map(lambda x: x[1], row_results)
        # train_rows_preds = map(lambda x: x[2], row_results)

        test_rows = filter(lambda x: x != None, test_rows)
        cv_rows = filter(lambda x: x != None, cv_rows)
        # train_rows_preds = filter(lambda x: x != None, train_rows_preds)

        # pdb.set_trace()
        test_preds = np.vstack(test_rows).astype(int)
        cv_preds = np.vstack(cv_rows).astype(int)
        # train_preds = np.vstack(train_rows_preds).astype(int)

        print test_preds.shape, 'test preds shape'
        print cv_preds.shape, 'cv preds shape'
        # print train_preds.shape, 'train preds shape'
        print test_preds[0], 'first row of test preds'
        print cv_preds[0], 'first row of cv preds'
        # print train_preds[0], 'first row of train preds'

        sorted_test = test_preds[test_preds[:, 0].argsort()]
        print "saving top t test preds"
        np.savetxt(folder + 'test_top_t.csv' , sorted_test,\
            fmt = '%s', delimiter = ',')

        # sorted_train = train_preds[train_preds[:, 0].argsort()]
        # print "saving top t train preds"
        # np.savetxt(submission_file + '_train_top_t', sorted_train,\
        #    fmt = '%s', delimiter = ',')

        sorted_cv = cv_preds[cv_preds[:, 0].argsort()]
        print "saving top t cv preds"
        np.savetxt(folder + 'cv_top_t.csv' , sorted_cv,\
            fmt = '%s', delimiter = ',')

        actual_cv = cv_data[:, -1].astype(int).reshape(-1, 1)
        cv_a_p = np.hstack((sorted_cv, actual_cv))
        apk_list = map(lambda row: apk(row[-1:], row[1:4]), cv_a_p)
        self.cv_mean_precision = np.mean(apk_list)
        print "mean precision of cross validation set", str(self.cv_mean_precision)

        sorted_test = sorted_test.astype(str)
        submission = open(submission_file, 'wb')
        submission.write('row_id,place_id\n')
        for i in range(len(sorted_test)):
            row = sorted_test[i]
            row_id = row[0]
            row_prediction_string = ' '.join(row[1:4])
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


