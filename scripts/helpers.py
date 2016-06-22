import csv
import matplotlib.pyplot as plt
import datetime
import numpy as np
import random
from tqdm import tqdm
from scipy import stats

start_date = datetime.datetime(2016, 5, 1)

days = lambda t: (int(t) % (86400*7))/ 86400
hours = lambda t: (int(t) % 86400) / 3600
quarter_days = lambda t: (int(t) % 86400) / (3600 * 4)

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

def generate_cv_and_training_data_sets(input_file, cv_file, train_file,\
    cutoff = 0.02, cv_count_cutoff = 5):
    """
    generate cross validation and training data set.
    input_file: initial training_file
    cv_file: output cross_validation file
    train_file: output_training file
    cutoff: random.random < cutoff => datapoint goes to cv set
    cv_count_cutoff: only if that place has more count than the cutoff it goes to cv set
    """

    # build the place_id matrix
    file_names = map(lambda x: '../split_train/' + str(x) + '_place', range(1, 23))
    use_cols = (0, 1)
    # load_data = lambda file_name: np.loadtxt(file_name, dtype = 'float', delimiter = ',',\
    #     skiprows = 1, usecols = use_cols)
    # sub_matrices = map(load_data, file_names)
    # count_matrix = np.vstack(sub_matrices).astype(int)
    cards = np.loadtxt('../place_ids_cardinality.txt', dtype = int)
    cards_trans = map(lambda x: (x[1]. x[0]), cards)
    count_dict = dict(cards_trans)
    del(sub_matrices)
    f = open(input_file, 'rb')
    fcsv = csv.reader(f)
    suffix = '_' + str(cutoff) + '_' + str(cv_count_cutoff) + '.csv'
    c = open(cv_file + suffix, 'wb')
    t = open(train_file + suffix, 'wb')

    while True:
        try:
            a = fcsv.next()
            r = random.random()
            place_count = count_dict[int(a[-1])]
            # if len(place_count) > 0:
            #     count = place_count[0][1]
            # else:
            #     count = 0
            if (random.random() < cutoff) and (place_count > cv_count_cutoff):
                c.write(','.join(a) + '\n')
            else:
                t.write(','.join(a) + '\n')
        except Exception, e:
            print e
            break
    f.close()
    c.close()
    t.close()


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

def compute_mode_xy(file_name, output_file_name):
    """
    compute mode of x, y on a sorted train chunk.
    also for x and y make the round down the values by one decimal point,
    for example, x = 2.3456 => x ~ 2.345, same for y
    """
    data = load_data(file_name)
    sorted_data = data[data[:, 4].argsort()]
    place_ids = np.unique(data[:, 4])
    place_wise_data = np.zeros((len(place_ids), 5))
    print "Iterating through each place and computing mode"
    for i, place_id in tqdm(enumerate(place_ids)):
        temp_data_set = sorted_data[sorted_data[:, 4] == place_id]
        xarr = temp_data_set[:, 0]
        yarr = temp_data_set[:, 1]
        xarr = xarr.astype('float')
        yarr = yarr.astype('float')
        xarrr = np.around(xarr, 3)
        yarrr = np.around(yarr, 3)
        x_mode = stats.mode(xarrr)[0][0]
        y_mode = stats.mode(yarrr)[0][0]
        xarrr = np.around(xarr, 2)
        yarrr = np.around(yarr, 2)
        x_mode_2 = stats.mode(xarrr)[0][0]
        y_mode_2 = stats.mode(yarrr)[0][0]
        mode_stats = np.array([place_id, x_mode, y_mode, x_mode_2, y_mode_2])
        place_wise_data[i] = mode_stats

    print "Saving File"
    np.savetxt(output_file_name, place_wise_data, delimiter = ',',\
        fmt = ['%.0f', '%.3f', '%.3f', '%.2f', '%.2f'],\
        header = 'place_id, x_mode_3, y_mode_3, x_mode_2, y_mode_2')
    return

# dumb model is a sample model that returns three given place ids,
# no matter what the test data point is
def dumb_model(x):
    print "inside dumb model", x
    return ['8523065625', '1757726713', '1137537235']

class BaseModel(object):
    cross_validation_file = '../cross_validation_02.csv'
    test_file = '../test.csv'
    cv_mean_precision = 0.0

    def __init__(self, cross_validation_file = '../cross_validation_02.csv', \
        test_file = '../test.csv', description = 'test run'):
        self.cross_validation_file = cross_validation_file
        self.test_file = test_file
        self.description = description

    def get_place_id(self, row):
        """
        get_place_id :: [x, y, accuracy, time] -> [place1, place2, place3]
        """
        return ['8523065625', '1757726713', '1137537235']

    def get_place_ids(self, rows):
        """
        vectorized form for get_place_id
        """
        pass

    def test_get_place_id(self):
        row = ['0.1675', '1.3608', '107', '930883']
        print "row is => ", row
        place_ids = self.get_place_id(row)
        print "Place ids are ", place_ids

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
                print row
                print e
                break
        cv_mean_precision = total_avg_precision/n
        self.cv_mean_precision = cv_mean_precision
        f.close()
        return cv_mean_precision


