

import grid_generation as grid
from nearest_distance_from_mean_place_ids import build_stat_xy_matrix, build_time_matrix
import numpy as np
import os
from tqdm import tqdm
import itertools
from multiprocessing import Pool
from base_scikit_learn_model import get_grids_of_a_point
import csv
import traceback

g = grid.Grid(200, 50, 20, 5, pref = 'grid')

def get_stat_dict(matrix):
    return dict(map(lambda x: [int(x[0]), x[1:]], matrix))

def load_data_from_grid(m, n):
    f = g.getGridFile(m, n)
    return np.loadtxt(f, delimiter = ',')

def get_time_stat(time_dict, id):
    if id in time_dict:
        return time_dict[id]
    else:
        return [-999]*8

def get_place_stat(place_stats_dict, id):
    if id in place_stats_dict:
        return place_stats_dict[id]
    else:
        return [-999]*4

submission_name = 'ec2_colsample_bytree0_6_scale_pos_weight1_min_child_weight6_subsample0_9_eta0_1_alpha0_005_max_depth3_gamma0_1_th3_n200'
place_file_names = map(lambda x: '../split_train/' + str(x) + '_place', range(1, 23))
time_file_names = map(lambda x: '../split_train/' + str(x) + '_time', range(1, 23))
load_data = lambda file_name, use_cols: np.loadtxt(file_name, dtype = 'float', delimiter=',', skiprows=1, usecols = use_cols)

place_stats_dict = get_stat_dict(np.vstack(map(lambda f: load_data(f, (0, 1, 4, 5, 6)), place_file_names)))
#place_stats_dict[place_id] = [count, a_median, x_median, y_median]

time_dict = get_stat_dict(np.vstack(map(lambda f: load_data(f, (0, 1, 2, 3, 4, 5, 6, 7, 8)), time_file_names)))
#time_dict[place_id] = [hour, hour_2, hour_3, hour_4, hour_6, hour_8, weekday, month]

def get_time_stat(id):
    if id in time_dict:
        return time_dict[id]
    else:
        return [-999]*8

def get_place_stat(id):
    if id in place_stats_dict:
        return place_stats_dict[id]
    else:
        return [-999]*4

def get_features_from_ids(xyat, ids):
    return np.concatenate(map(lambda id: get_feature_from_id(xyat, id), ids))

def get_feature_from_id(xyat, id):
    f = get_time_stat(id)
    p = get_place_stat(id)
    abd = abs_dt(xyat[:2], p[-2:])
    return np.hstack((f, p, abd))

def abs_dt(p1, p2):
    return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]) 

def get_top_places_file(g, m, n, submission_name):
    return g.getFolder()[:-1] + '_' + submission_name + '/' + '_'.join(['top_t_preds', str(m), str(n)]) + '.csv'

def generate_features(grid, submission_name):
    folder = grid.getFeaturesFolder(submission_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    # for m in range(grid.max_m + 1):
    for m in range(1):
        # for n in range(grid.max_n + 1):
        for n in range(1):
            generate_feature_in_grid(grid, submission_name, m, n)

class GridLoader(object):
    def __init__(self, grid, submission_name):
        self.grid = grid
        self.submission_name = submission_name

    def __call__(self, grid_id):
        generate_feature_in_grid(self.grid, self.submission_name, grid_id[0], grid_id[1])


fmt = ['%.0f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f',\
    '%.1f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.0f', '%.2f', '%.4f', '%.4f', '%.4f']

def generate_test_feature_files(grid, submission_name):
    test_feature_prefix = grid.getFeaturesFolder(submission_name) + 'test_feature'
    test_features = np.loadtxt(test_feature_prefix + '.csv', delimiter = ',')

    test_data = np.loadtxt('../test.csv', dtype = float, delimiter = ',')
    test_feature_data = np.loadtxt(grid.getFeaturesFolder(submission_name) + \
        'test_feature.csv', delimiter = ',')
    test_combined = np.hstack((test_data, test_feature_data[:, 1:]))
    del(test_data)
    del(test_feature_data)

    test_grid_wise_data = [[[] for n in range(grid.max_n + 1)]\
            for m in range(grid.max_m + 1)]


    print "converting test data to grid wise"
    for i in tqdm(range(len(test_combined))):
        m, n = get_grids_of_a_point((test_combined[i][1], test_combined[i][2]), grid)[0]
        test_grid_wise_data[m][n].append(test_combined[i])

    for m in range(grid.max_m + 1):
        for n in range(grid.max_n + 1):
            np.savetxt('_'.join([test_feature_prefix, str(m), str(n)]) + '.csv', \
                np.array(test_grid_wise_data[m][n]), delimiter = ',', \
                fmt = ['%.0f', '%.5f', '%.5f', '%.0f', '%.0f'] + fmt[1:])

def generate_test_feature_files(grid, submission_name):
    test_feature_prefix = grid.getFeaturesFolder(submission_name) + 'test_feature'
    test_data = open('../test.csv', 'rb')
    test_data_csv = csv.reader(test_data)

    test_feature_data = open(test_feature_prefix + '.csv', 'rb')
    test_feature_csv = csv.reader(test_feature_data)

    file_handles = [[open('_'.join([test_feature_prefix, str(m), str(n)]), 'wb')\
                for m in range(grid.max_n + 1)]\
                for n in range(grid.max_m + 1)]

    progress = 0
    while True:
        try:
            a = test_data_csv.next()
            b = test_feature_csv.next()
            line = ','.join(b) + '\n'
            c = (float(a[1]), float(a[2]))
            m, n = get_grids_of_a_point(c, grid)[0]
            progress += 1
            file_handles[m][n].write(line)
            if progress % 1000000 == 0:
                print 'parsing line %s' %(progress)
        except Exception, e:
            print e
            print a
            traceback.format_exc()
            break

    temp = [map(lambda file_handle: file_handle.close(), row) for row in file_handles]


def generate_feature_in_grid(grid, submission_name, m, n):
    """
    """
    data = load_data_from_grid(m, n)
    if len(data) == 0:
        return
    try:
        top_t_data = np.loadtxt(get_top_places_file(g, m, n, submission_name), delimiter = ',', dtype = int)
    except IOError:
        print m, n, "no top_t_data file exists for this grid cell"
        top_t_data = np.hstack((data[:, 1].reshape(-1, 1), [[123]*10]*len(data)))

    data_dict = get_stat_dict(data)
    top_t_dict = get_stat_dict(top_t_data)

    features = {}
    for i in data_dict.keys():
        if i in top_t_dict:
            features[i] = get_features_from_ids(data_dict[i], top_t_dict[i])
        else:
            features[i] = np.array([-999]*130)

    feature_items = sorted(features.items(), cmp = lambda x, y: cmp(x[0], y[0]))
    feature_data = np.array(map(lambda row: np.hstack(([row[0]], row[1])), feature_items))
    file_name = grid.getFeaturesFolder(submission_name) + '_'.join(['feature', str(m), str(n)]) + '.csv'
    np.savetxt(file_name, feature_data, delimiter = ',', fmt = fmt)

def generate_grid_features(grid, submission_name):
    m = range(grid.max_m + 1)
    n = range(grid.max_n + 1)
    grid_ids = itertools.product(m, n)
    p = Pool(16)
    result = p.map(GridLoader(grid, submission_name), grid_ids)
    p.close()
    p.join()

def generate_test_feature(grid, submission_name):
    folder = grid.getFeaturesFolder(submission_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    orig_file = '../test.csv'
    top_t_file = grid.getTopPlacesFolder(submission_name) + 'test_top_t.csv'
    feature_file = grid.getFeaturesFolder(submission_name) + 'test_feature.csv'
    generate_feature(orig_file, top_t_file, feature_file)

def generate_cv_feature(grid, submission_name):
    folder = grid.getFeaturesFolder(submission_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    orig_file = '../main_cv_0.02_5.csv'
    top_t_file = grid.getTopPlacesFolder(submission_name) + 'cv_top_t.csv'
    feature_file = grid.getFeaturesFolder(submission_name) + 'cv_feature.csv'
    generate_feature(orig_file, top_t_file, feature_file)

def generate_feature(orig_file, top_t_file, feature_file):
    data = np.loadtxt(orig_file, delimiter = ',')
    top_t_data = np.loadtxt(top_t_file, delimiter = ',', dtype = int)

    data_dict = get_stat_dict(data)
    top_t_dict = get_stat_dict(top_t_data)

    features = {}
    for i in tqdm(data_dict.keys()):
        if i in top_t_dict:
            features[i] = get_features_from_ids(data_dict[i], top_t_dict[i])
        else:
            features[i] = np.array([-999]*130)

    feature_items = sorted(features.items(), cmp = lambda x, y: cmp(x[0], y[0]))
    feature_data = np.array(map(lambda row: np.hstack(([row[0]], row[1])), feature_items))

    np.savetxt(feature_file, feature_data, delimiter = ',', fmt = fmt)
