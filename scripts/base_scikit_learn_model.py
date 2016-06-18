import os
import pickle
import time

import numpy as np

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

from helpers import BaseModel, days, hours
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

    def __init__(self, grid = base_grid, threshold = 20):
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
            y_transformer = {'label_encoder': label_encoder}
        new_y = y_transformer['label_encoder'].transform(y).reshape(-1, 1)
        return (new_y, y_transformer)

    def train(self):
    """
    training each grid seperately and storing building the self.model parameter
    the dictionary model[m][n] contains details about the grid(m, n)
    """
        for m in range(self.grid.max_m + 1):
            for n in range(self.grid.max_n + 1):
                # train each grid seperately
                self.train_grid(m, n)

    def train_grid(self, m, n):
    """
    """
        print "Training %s, %s grid" %(m, n)
        init_time = time.time()
        data = np.loadtxt(grid.getGridFile(m, n), dtype = float, delimiter = ',')
        mask = np.array(map(lambda x: self.grid.M[0][0][x] > 20, data[:, 5]))
        masked_data = data[mask, :]
        X, x_transformer = self.transform_x(masked_data[:, (1, 2, 3, 4)])
        Y, y_transformer = self.transform_y(masked_data[:, 5])

        model[m][n]['x_transformer'] = x_transformer
        model[m][n]['y_transformer'] = y_transformer

        model[m][n]['model'] = KNeighborsClassifier(n_neighbors = 29,\
            weights = 'distance', metric = 'manhattan')
        model[m][n]['model'].fit(X, Y)

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
        for i in range(len(test_data)):
            m, n = get_grids_of_a_point((test_data[i][1], test_data[i][2]), self.grid)[0]
            grid_wise_data[m][n].append(test_data[i])

        predictions = []
        for m in range(self.grid.max_m + 1):
            for n in range(self.grid.max_n + 1):
                predictions.append(self.predict_grid(grid_wise_data[m][n], m, n))

        predictions = np.vstack(tuple(predictions))
        sorted_row_predictions = predictions[predictions[:, 0].argsort()]
        return sorted_row_predictions

    def predict_grid(self, grid_data, m, n):
        """
        grid_data is test/cv data from that particular grid
        return row_id, and top 3 predictions
        """
        grid_data = np.array(grid_data)
        temp_x = self.transform_x(grid_data[:, (1, 2, 3, 4)])
        prediction_probs = self.model[m][n]['model'].predict_proba(temp_x)
        top_3_placeids = self.model[m][n]['y_transformer']['label_encoder'].inverse_transform(np.argsort(prediction_probs, axis=1)[:,::-1][:,:3])
        return np.hstack((grid_data[:, 0].reshape(-1, 1), top_3_placeids))
