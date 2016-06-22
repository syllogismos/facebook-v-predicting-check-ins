


from base_scikit_learn_model import SklearnModel
from grid_generation import Grid
from helpers import days, hours, quarter_days
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

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
