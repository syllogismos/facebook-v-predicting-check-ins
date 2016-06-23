


from base_scikit_learn_model import SklearnModel
from grid_generation import Grid, generate_grid_wise_cardinality_and_training_files
from grid_generation import get_top_3_places_of_dict, get_grids
from helpers import days, hours, quarter_days
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

import numpy as np
import os
import pickle

grid = Grid(X = 800, Y = 400, xd = 200, yd = 100, pref = 'grid')


class KNN(SklearnModel):

    def transform_x(self, X, x_transformer = None):
        days_v = np.array(map(days, X[:, 3])).reshape(-1, 1)
        quarter_days_v = np.array(map(quarter_days, X[:, 3])).reshape(-1, 1)

        if x_transformer == None:
            scaler = preprocessing.StandardScaler().fit(X[:, (0, 1, 2)])
            days_enc = preprocessing.OneHotEncoder()
            days_enc.fit(days_v)
            quarter_days_enc = preprocessing.OneHotEncoder()
            quarter_days_enc.fit(quarter_days_v)
            x_transformer = {}
            x_transformer['scaler'] = scaler
            x_transformer['days_enc'] = days_enc
            x_transformer['quarter_days_enc'] = quarter_days_enc
        else:
            scaler = x_transformer['scaler']
            days_enc = x_transformer['days_enc']
            quarter_days_enc = x_transformer['quarter_days_enc']

        xya = scaler.transform(X[:, (0, 1, 2)])
        days_oh = days_enc.transform(days_v).toarray()
        quarter_days_oh = quarter_days_enc.transform(quarter_days_v).toarray()

        new_X = np.hstack((xya, days_oh, quarter_days_oh))

        return (new_X, x_transformer)

    def custom_classifier(self, X, Y):
        clf = KNeighborsClassifier(n_neighbors = 7, weights = 'distance', metric = 'manhattan')
        clf.fit(X, np.ravel(Y))
        return clf

class KNN_feature_weights(SklearnModel):

    def transform_x(self, X, x_transformer = None):
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
        weekday_v = (weekday_v%7 + 1)*fw[3]
        month_v = (month_v%12 +1)*fw[4]
        accuracy_v = np.log10(X[:, 2])*fw[6]
        x_v = X[:, 0]*fw[0]
        y_v = X[:, 1]*fw[1]
        new_X = np.hstack((x_v.reshape(-1, 1),\
                         y_v.reshape(-1, 1),\
                         accuracy_v.reshape(-1, 1),\
                         hour_v.reshape(-1, 1),\
                         weekday_v.reshape(-1, 1),\
                         month_v.reshape(-1, 1),\
                         year_v.reshape(-1, 1)))
        return (new_X, x_transformer)


    def custom_classifier(self, X, Y):
        clf = KNeighborsClassifier(n_neighbors = 16, weights = 'distance', metric = 'manhattan')
        clf.fit(X, np.ravel(Y))
        return clf


class top_3_using_knn_model(SklearnModel):
    def __init__(self, cv_file = '../main_cv_0.02_5.csv', \
        test_file = '../test.csv', X = 14, Y = 10, xd = 4, yd = 1, pref = 'grid'):

        self.cross_validation_file = cv_file
        self.test_file = test_file
        self.X = X
        self.Y = Y
        self.xd = xd
        self.yd = yd
        self.pref = pref
        self.description = 'top_3_grid_places_model_' + str(X) + '_' + \
            str(Y) + '_' + \
            str(xd) + '_' + \
            str(yd) + '_'
        print "@@@@@@@@@@@@@@@@@@@"
        print self.description

        # load cardinality matrix and generate files if the run is not done for above X,Y
        # values
        folder_name = '../' + '_'.join([pref, str(X), str(Y), str(xd), str(yd)]) + '/'
        status = folder_name + 'status.pkl'
        if not os.path.exists(status):
            print "Generating grid wise files"
            generate_grid_wise_cardinality_and_training_files('../main_train_0.02_5.csv', \
                self.X, self.Y, self.xd, self.yd, self.pref)
        print "Loading cardinality matrix from pickle"
        self.M = pickle.load(open(folder_name + 'cardinality_pickle.pkl', 'rb'))
        self.M = [map(get_top_3_places_of_dict, row) for row in self.M]


    def get_place_id(self, row):
        c = (float(row[1]), float(row[2]))
        m, n = get_grids(c, self.X, self.Y, self.xd, self.yd)[0]
        pred = map(int, self.M[m][n])
        y = len(pred)
        if y < 3:
            temp = [5348440074]*(3-y)
            pred += temp
        return pred

    # def predict(self, test_data):
    #     preds = []
    #     for i in range(len(test_data)):
    #         i_place_ids = self.get_place_id(test_data[i])
    #         preds.append([int(test_data[i][0])] + i_place_ids)
    #     print preds[:3]
    #     return np.array(preds)

    def predict_grid(self, grid_data, m, n):
        print "predicting grid %s, %s" %(m, n)
        grid_data = np.array(grid_data)
        preds = []
        for i in range(len(grid_data)):
            i_place_ids = self.get_place_id(grid_data[i])
            preds.append(i_place_ids)
        preds = np.array(preds)
        ret = np.hstack((grid_data[:, 0].reshape(-1, 1), preds))
        # print ret[:3]
        return ret

    def predict(self, test_data):
        """
        test_data is a matrix whose row contains
        row_id, x, y, a, time, (place_id)
        place_id is optional
        """
        max_m, max_n = get_grids((10.0, 10.0), self.X, self.Y, self.xd, self.yd)[0]
        grid_wise_data = [[[] for n in range(max_n + 1)]\
            for m in range(max_m + 1)]
        print "Computing grid_wise_data from test_data"
        for i in range(len(test_data)):
            m, n = get_grids((float(test_data[i][1]), float(test_data[i][2])), self.X, self.Y, self.xd, self.yd)[0]
            grid_wise_data[m][n].append(test_data[i])

        predictions = []
        for m in range(max_m + 1):
            for n in range(max_n + 1):
                if len(grid_wise_data[m][n]) > 0:
                    predictions.append(self.predict_grid(np.array(grid_wise_data[m][n]), m, n))

        predictions = np.vstack(tuple(predictions)).astype(int)
        sorted_row_predictions = predictions[predictions[:, 0].argsort()]
        return sorted_row_predictions



if __name__ == '__main__':
    run = 100
    # days one hot encoder, quarter days one hot encoder, xya scaled
    g = Grid(X = 800, Y = 400, xd = 200, yd = 100, pref = 'grid')
    knn = KNN(grid = g, threshold = 7, description = 'days_oh_quarter_days_oh_scaled_n_neighbors7_xya')
    f = open('../cv_run_21_Jun.txt', 'ab')
    f.write(str(run) + '\n')
    f.write(knn.description)
    print "Traning the classifier"
    knn.train()
    cv = knn.check_cross_validation()
    f.write(str(cv))
    f.write('\n')
    knn.generate_submission_file('../knn_21_june_oh_encoding_n_neightbors7_100')
    f.close()
