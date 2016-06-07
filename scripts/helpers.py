import csv
import matplotlib.pyplot as plt

def plot_train_data(file_name):
    f = open(file_name, 'rb')
    fcsv = csv.reader(f)
    xys = map(lambda x: (float(x[1]), float(x[2])), fcsv)
    plt.plot(map(lambda x: x[0], xys), map(lambda x: x[1], xys), 'ro')
    plt.axis([0,10,0,10])
    plt.show()
