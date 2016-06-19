import os
import pickle
import time

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from helpers import BaseModel, days, hours, apk
from grid_generation import Grid, get_grids

base_grid = Grid(X = 800, Y = 400, xd = 200, yd = 100, pref = 'grid')

def get_grids_of_a_point(c, grid, buff = False):
    """
    c :: (x, y)
    grid :: Grid
    buff :: Boolean
    given a point and grid defn return the grid it falls into
    if buff True, returns other grids the point might falls into cause of buffers
    """
    return get_grids(c, grid.X, grid.Y, grid.xd, grid.yd, train = buff)

class SklearnModel(BaseModel):

    def __init__(self, cross_validation_file = '../main_cv_0.02_5.csv',\
        test_file = '../test.csv', grid = base_grid, threshold = 20,\
        description = 'test_sklearn_model'):
        self.cross_validation_file = cross_validation_file
        self.test_file = test_file
        self.description = description
        self.grid = grid
        self.grid.generateCardinalityMatrix()
        self.threshold = threshold
        self.model = [[{} for t in range(self.grid.max_n + 1)]\
            for s in range(self.grid.max_m + 1)]

    def transform_x(self, X, x_transformer = None):
        """
        X :: [[x, y, a, time]], numpy array float
        returns a tuple of transformed X and dict that contains details
        that help transform test data
        """
        days_v = np.array(map(days, X[:, 3])).reshape(-1, 1)
        hours_v = np.array(map(days, X[:, 3])).reshape(-1, 1)
        new_X = np.hstack((X[:, (0, 1, 2)], days_v, hours_v))
        x_transformer = {}
        return (new_X, x_transformer)

    def transform_y(self, y, y_transformer = None):
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

    def custom_classifier(self, X, Y):
        """
        Given X, Y
        define the model, and train it
        """
        if self.grid.pref == 'test':
            clf = KNeighborsClassifier(n_neighbors = 2,\
                weights = 'distance', metric = 'manhattan')
        else:
            clf = KNeighborsClassifier(n_neighbors = 29,\
                weights = 'distance', metric = 'manhattan')
        clf.fit(X, np.ravel(Y))
        return clf

    def train(self, force = False):
        """
        training each grid seperately and storing building the self.model parameter
        the dictionary model[m][n] contains details about the grid(m, n)

        using force flag we can recompute the model even if pickle exists from earlier train run
        """
        trained_model_file = self.grid.getFolder() + self.description + '_' +\
            'grid_model_pickle.pkl'

        # if (not os.path.exists(trained_model_file)) or force:
        #     for m in range(self.grid.max_m + 1):
        #         for n in range(self.grid.max_n + 1):
        #             # train each grid seperately
        #             self.train_grid(m, n)
        #     pickle.dump(self.model, open(trained_model_file, 'wb'))
        # else:
        #     self.model = pickle.load(open(trained_model_file), 'rb')

        for m in range(self.grid.max_m + 1):
            for n in range(self.grid.max_n + 1):
                # train each grid seperately
                self.train_grid(m, n)

    def train_grid(self, m, n):
        """
        Helper function for train function that takes the grid cell to be trained
        """
        print "Training %s, %s grid" %(m, n)
        init_time = time.time()
        data = np.loadtxt(self.grid.getGridFile(m, n), dtype = float, delimiter = ',')
        if len(data) == 0:
            # if the grid contains barely zero data, make model None
            self.model[m][n]['model'] = None
            return
        mask = np.array(map(lambda x: self.grid.M[m][n][x] > self.threshold, data[:, 5]))
        masked_data = data[mask, :]
        X, x_transformer = self.transform_x(masked_data[:, (1, 2, 3, 4)])
        Y, y_transformer = self.transform_y(masked_data[:, 5])

        self.model[m][n]['x_transformer'] = x_transformer
        self.model[m][n]['y_transformer'] = y_transformer

        if len(Y) == 0:
            # if masked data is of length zero then also make model None
            self.model[m][n]['model'] = None
        else:
            self.model[m][n]['model'] = self.custom_classifier(X, Y)

        print "Time taken to train grid %s, %s is: %s" %(m, n, time.time() - init_time)
        print "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"


    def predict(self, test_data):
        """
        test_data is a matrix whose row contains
        row_id, x, y, a, time, (place_id)
        place_id is optional
        """
        grid_wise_data = [[[] for n in range(self.grid.max_n + 1)]\
            for m in range(self.grid.max_m + 1)]
        print "Computing grid_wise_data from test_data"
        for i in range(len(test_data)):
            m, n = get_grids_of_a_point((test_data[i][1], test_data[i][2]), self.grid)[0]
            grid_wise_data[m][n].append(test_data[i])

        predictions = []
        for m in range(self.grid.max_m + 1):
            for n in range(self.grid.max_n + 1):
                if len(grid_wise_data[m][n]) > 0:
                    predictions.append(self.predict_grid(np.array(grid_wise_data[m][n]), m, n))

        predictions = np.vstack(tuple(predictions)).astype(int)
        sorted_row_predictions = predictions[predictions[:, 0].argsort()]
        return sorted_row_predictions

    def predict_grid(self, grid_data, m, n):
        """
        grid_data is test/cv data from that particular grid
        return row_id, and top 3 predictions
        """
        print "predicting grid %s, %s" %(m, n)
        grid_data = np.array(grid_data)
        temp_x = self.transform_x(grid_data[:, (1, 2, 3, 4)])[0]
        if self.model[m][n]['model'] == None:
            top_3_placeids = np.array([[5348440074, 9988088517, 4048573921]]*len(temp_x))
        else:
            prediction_probs = self.model[m][n]['model'].predict_proba(temp_x)
            top_3_placeids = self.model[m][n]['y_transformer']['encoder'].inverse_transform(np.argsort(prediction_probs, axis=1)[:,::-1][:,:3])

            # temporary hack when the no of predictions for a row is less than 3
            x, y = top_3_placeids.shape
            if y < 3:
                temp_array = np.array([[5348440074]*(3-y)]*len(top_3_placeids))
                top_3_placeids = np.hstack((top_3_placeids, temp_array))

        return np.hstack((grid_data[:, 0].reshape(-1, 1), top_3_placeids))

    def generate_submission_file(self, submission_file):
        """
        Generate the submission file using the trained model in each indivudual grid
        """
        test_data = np.loadtxt(self.test_file, dtype = float, delimiter = ',')
        predictions = self.predict(test_data)
        predictions = predictions.astype(int)
        predictions = predictions.astype(str)
        submission = open(submission_file, 'wb')
        submission.write('row_id,place_id\n')
        for i in range(len(predictions)):
            row = predictions[i]
            row_id = row[0]
            row_prediction_string = ' '.join(row[1:])
            submission.write(row_id + ',' + row_prediction_string + '\n')
            if i % 1000000 == 0:
                print "Generating %s row of test data" %(i)
        submission.close()

    def check_cross_validation(self):
        """
        Compute cross validation of the trained model on the cv file that we initialized
        """
        data = np.loadtxt(self.cross_validation_file, dtype = float, delimiter = ',')
        predictions = self.predict(data)
        predictions = predictions.astype(int)
        actual = data[:, -1].astype(int).reshape(-1, 1)
        preds = np.hstack((predictions, actual))
        apk_list = map(lambda row: apk(row[-1:], row[1: -1]), preds)
        self.cv_mean_precision = np.mean(apk_list)
        return self.cv_mean_precision
