import csv
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
from tqdm import tqdm

start_date = datetime.datetime(2016, 5, 1)

def apk(actual, predicted, k=3):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def plot_train_data(file_name):
    f = open(file_name, 'rb')
    fcsv = csv.reader(f)
    xys = map(lambda x: (float(x[1]), float(x[2])), fcsv)
    plt.plot(map(lambda x: x[0], xys), map(lambda x: x[1], xys), 'ro')
    plt.axis([0,10,0,10])
    plt.show()

def sample_file(input_file, output_file, cutoff):
    """
    Sample input file randomly with a given cutoff
    and create a local test data set.
    """
    f = open(input_file, 'rb')
    o = open(output_file, 'wb')
    while True:
        try:
            a = f.next()
            if random.random() < cutoff:
                o.write(a)
        except Exception, e:
            break
    o.close()
    f.close()

def load_data(file_name):
    f = open(file_name, 'rb')
    fcsv = csv.reader(f)
    data = map(lambda x: [float(x[1]),
                          float(x[2]),
                          int(x[3]),
                          start_date + datetime.timedelta(seconds = int(x[4])),
                          int(x[5])
                         ],
               fcsv)
    return np.array(data)


def compute_place_wise_stats(file_name, output_file_name):
    """
    Compute place wise stats on a sorted training file chunk
    and create a csv file that contains, mean, median, std, var of x, y and accuracy
    """
    data = load_data(file_name)
    sorted_data = data[data[:, 4].argsort()]
    place_ids = np.unique(data[:, 4])
    place_wise_data = np.zeros((len(place_ids), 14))
    print "Iterating through each place and computing stats"
    for i, place_id in tqdm(enumerate(place_ids)):
        temp_data_set = sorted_data[sorted_data[:, 4] == place_id]
        xya = temp_data_set[:, 0:3]
        xya = xya.astype('float')
        no_of_events = len(xya)
        var = np.var(xya, axis = 0)
        mean = np.mean(xya, axis = 0)
        median = np.median(xya, axis = 0)
        std = np.std(xya, axis = 0)
        var = np.var(xya, axis = 0)
        stats = np.concatenate((np.array([place_id, no_of_events]), mean, median, std, var), axis = 0)
        place_wise_data[i] = stats

    print "Saving File"
    np.savetxt(output_file_name, place_wise_data, delimiter = ',',\
        fmt = ['%.0f', '%.0f',
        '%.10f', '%.10f', '%.10f',\
        '%.10f', '%.10f', '%.10f',\
        '%.10f', '%.10f', '%.10f',\
        '%.10f', '%.10f', '%.10f'],\
        header = 'place_id, count, x_mean, y_mean, a_mean, x_median, y_median, a_median, x_std, y_std, a_std, x_var, y_var, a_var')
    return


def build_mean_xy_matrix():
    file_names = map(lambda x: '../split_train/' + str(x) + '_place', range(1,23))
    load_data = lambda file_name: np.loadtxt(file_name, dtype = 'float', delimiter=',', skiprows=1, usecols=(0, 2, 3))
    sub_matrices = map(load_data, file_names)
    mean_matrix = np.vstack(sub_matrices)
    del(sub_matrices)
    return mean_matrix

# dumb model is a sample model that returns three given place ids,
# no matter what the test data point is
def dumb_model(x):
    print "inside dumb model", x
    return ['8523065625', '1757726713', '1137537235']

class Model:
    cross_validation_file = '../cross_validation_02.csv'
    test_file = '../test.csv'
    cv_mean_precision = 0.0

    def __init__(self, cross_validation_file = '../cross_validation_02.csv', test_file = '../test.csv'):
        self.cross_validation_file = cross_validation_file
        self.test_file = test_file

    def get_place_id(self, row):
        """
        get_place_id :: [x, y, accuracy, time] -> [place1, place2, place3]
        """
        return ['8523065625', '1757726713', '1137537235']

    def train_model(self):
        pass

    def generate_submission_file(self, submission_file):
        test = open(self.test_file, 'rb')
        test_csv = csv.reader(test)
        submission = open(submission_file, 'wb')
        header = 'row_id,place_id'
        submission.write(header + '\n')
        n = 0
        while True:
            try:
                row = test_csv.next()
                row_id = row[0]
                row_inp = row[1:]
                row_prediction = self.get_place_id(row_inp)
                row_prediction_string = ' '.join(row_prediction)
                row_output = row_id + ',' + row_prediction_string + '\n'
                submission.write(row_output)
                n += 1
                if n % 1000000 == 0:
                    print "Generating %s row of submission file" %(n)
            except:
                break
        submission.close()
        test.close()

    def get_cross_validation_mean_precision(self):
        total_avg_precision = 0
        n = 0
        f = open(self.cross_validation_file, 'rb')
        fcsv = csv.reader(f)
        while True:
            try:
                row = fcsv.next()
                row_place = row[-1:]
                row_inp = row[1:-1] # x, y, accuracy, time
                row_prediction = self.get_place_id(row_inp)
                row_apk = apk(row_place, row_prediction)
                n += 1
                total_avg_precision += row_apk
                if n % 10000 == 0:
                    print "computing cv precision for %s row" %(n)
            except Exception, e:
                print "Exception in get_cross_validation_error func"
                print e
                break
        cv_mean_precision = total_avg_precision/n
        self.cv_mean_precision = cv_mean_precision
        f.close()
        return cv_mean_precision


