import csv
import matplotlib.pyplot as plt
import datetime
import numpy as np
from tqdm import tqdm

start_date = datetime.datetime(2016, 5, 1)

def plot_train_data(file_name):
    f = open(file_name, 'rb')
    fcsv = csv.reader(f)
    xys = map(lambda x: (float(x[1]), float(x[2])), fcsv)
    plt.plot(map(lambda x: x[0], xys), map(lambda x: x[1], xys), 'ro')
    plt.axis([0,10,0,10])
    plt.show()


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
