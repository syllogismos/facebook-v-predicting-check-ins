

from helpers import BaseModel
import numpy as np
import math

def build_stat_xy_matrix(stat = 'mean', acc = 3):
    file_names = map(lambda x: '../split_train/' + str(x) + '_place', range(1,23))
    if stat == 'median':
        use_cols = (0, 5, 6)
    elif stat == 'mean':
        use_cols = (0, 2, 3)
    elif stat == 'mode':
        mode_matrix = build_mode_xy_matrix(acc = 3)
        return mode_matrix
    else:
        use_cols = (0, 2, 3)
    load_data = lambda file_name: np.loadtxt(file_name, dtype = 'float', delimiter=',', skiprows=1, usecols = use_cols)
    sub_matrices = map(load_data, file_names)
    mean_matrix = np.vstack(sub_matrices)
    del(sub_matrices)
    return mean_matrix

def build_mode_xy_matrix(acc = 3):
    file_names = map(lambda x: '../split_train/' + str(x) + '_mode', range(1,23))
    if acc == 2:
        use_cols = (0, 3, 4)
    elif acc == 3:
        use_cols = (0, 1, 2)
    else:
        use_cols = (0, 1, 2)
    load_data = lambda file_name: np.loadtxt(file_name, dtype = 'float', delimiter=',', skiprows=1, usecols = use_cols)
    sub_matrices = map(load_data, file_names)
    mode_matrix = np.vstack(sub_matrices)
    del(sub_matrices)
    return mode_matrix

def distance(a, b, distance_type = "sqrt"):
    """
    type can take `sqrt` or `abs`
    """
    if distance_type == 'sqrt':
        d = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    elif distance_type == 'abs':
        d = abs(a[0] - b[0]) + abs(a[1] - b[1])
    return d

class NearestToPlaces(BaseModel):

    def __init__(self, cross_validation_file = '../cross_validation_02.csv',\
        test_file = '../test.csv', stat = 'mean', distance_type = 'sqrt',\
        delta = 0.08, acc = 3):
        self.cross_validation_file = cross_validation_file
        self.test_file = test_file
        self.delta = delta
        self.distance_type = distance_type
        self.stat = build_stat_xy_matrix(stat = stat)
        self.stat_type = stat
        self.acc = acc

    def get_place_id(self, row):
        x = float(row[0])
        y = float(row[1])
        if self.stat_type == 'mode':
            x = np.around(x, self.acc)
            y = np.around(y, self.acc)

        closest_m = self.stat[(self.stat[:, 1] > (x - self.delta)) & \
                              (self.stat[:, 1] < (x + self.delta)) & \
                              (self.stat[:, 2] > (y - self.delta)) & \
                              (self.stat[:, 2] < (y + self.delta))]
        distances = map(lambda row: (row[0], distance((x, y), row[1:], \
                        distance_type = self.distance_type)), closest_m)
        sorted_places = map(lambda x: '%.10s'%x[0], sorted(distances, cmp = lambda x, y: cmp(x[1], y[1])))
        return sorted_places[:3]

if __name__ == '__main__1':
    """
    with delta 0.02 and find closest to mean and median positions.. closest by sqrt and abs
    """
    # f = open('../cv_results_run_1.txt', 'wb')

    # model_1 = NearestToPlaces(stat = 'mean', distance_type = 'sqrt', delta = 0.02)
    # model_1.generate_submission_file('../submission_nearest_place_mean_sqrt_0.02.csv')
    # cv_1 = model_1.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mean_sqrt_0.02.csv cv precision is %s \n' %(cv_1)
    # f.write(x)

    # model_2 = NearestToPlaces(stat = 'mean', distance_type = 'abs', delta = 0.02)
    # model_2.generate_submission_file('../submission_nearest_place_mean_abs_0.02.csv')
    # cv_2 = model_2.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mean_abs_0.02.csv cv precision is %s \n' %(cv_2)
    # f.write(x)

    # model_3 = NearestToPlaces(stat = 'median', distance_type = 'sqrt', delta = 0.02)
    # model_3.generate_submission_file('../submission_nearest_place_median_sqrt_0.02.csv')
    # cv_3 = model_3.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_median_sqrt_0.02.csv cv precision is %s \n' %(cv_3)
    # f.write(x)

    # model_4 = NearestToPlaces(stat = 'median', distance_type = 'abs', delta = 0.02)
    # model_4.generate_submission_file('../submission_nearest_place_median_abs_0.02.csv')
    # cv_4 = model_4.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_median_abs_0.02.csv cv precision is %s \n' %(cv_4)
    # f.write(x)

if __name__ == '__main__2':
    """
    do above runs with delta 0.08
    """
    f = open('../cv_results_run_2.txt', 'ab')

    # model_1 = NearestToPlaces(stat = 'mean', distance_type = 'sqrt', delta = 0.08)
    # model_1.generate_submission_file('../submission_nearest_place_mean_sqrt_0.02.csv')
    # cv_1 = model_1.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mean_sqrt_0.08.csv cv precision is %s \n' %(cv_1)
    # f.write(x)

    # model_2 = NearestToPlaces(stat = 'mean', distance_type = 'abs', delta = 0.02)
    # model_2.generate_submission_file('../submission_nearest_place_mean_abs_0.02.csv')
    # cv_2 = model_2.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mean_abs_0.02.csv cv precision is %s \n' %(cv_2)
    # f.write(x)

    # model_3 = NearestToPlaces(stat = 'median', distance_type = 'sqrt', delta = 0.08)
    # model_3.generate_submission_file('../submission_nearest_place_median_sqrt_0.08.csv')
    # cv_3 = model_3.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_median_sqrt_0.08.csv cv precision is %s \n' %(cv_3)
    # f.write(x)

    model_4 = NearestToPlaces(stat = 'median', distance_type = 'abs', delta = 0.08)
    model_4.generate_submission_file('../submission_nearest_place_median_abs_0.08.csv')
    cv_4 = model_4.get_cross_validation_mean_precision()
    x = 'submission_nearest_place_median_abs_0.08.csv cv precision is %s \n' %(cv_4)
    f.write(x)

    f.close()

if __name__ == '__main__3':
    """
    do runs using mode
    """
    f = open('../cv_results_run_4.txt', 'ab')

    # model_1 = NearestToPlaces(stat = 'mode', distance_type = 'sqrt', delta = 0.08, acc = 3)
    # model_1.generate_submission_file('../submission_nearest_place_mode_sqrt_0.08_3.csv')
    # cv_1 = model_1.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mode_sqrt_0.08_3.csv cv precision is %s \n' %(cv_1)
    # f.write(x)

    # model_2 = NearestToPlaces(stat = 'mode', distance_type = 'sqrt', delta = 0.08, acc = 2)
    # model_2.generate_submission_file('../submission_nearest_place_mode_sqrt_0.08_2.csv')
    # cv_2 = model_2.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mode_sqrt_0.08_2.csv cv precision is %s \n' %(cv_2)
    # f.write(x)

    # model_3 = NearestToPlaces(stat = 'mode', distance_type = 'abs', delta = 0.08, acc = 3)
    # model_1.generate_submission_file('../submission_nearest_place_mode_abs_0.08_3.csv')
    # cv_3 = model_3.get_cross_validation_mean_precision()
    # x = 'submission_nearest_place_mode_abs_0.08_3.csv cv precision is %s \n' %(cv_3)
    # f.write(x)

    model_4 = NearestToPlaces(stat = 'mode', distance_type = 'abs', delta = 0.08, acc = 2)
    model_4.generate_submission_file('../submission_nearest_place_mode_abs_0.08_2_copy.csv')
    cv_4 = model_4.get_cross_validation_mean_precision()
    x = 'submission_nearest_place_mode_abs_0.08_2.csv cv precision is %s \n' %(cv_4)
    f.write(x)

    f.close()
