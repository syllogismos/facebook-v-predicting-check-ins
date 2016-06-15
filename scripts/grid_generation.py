import os
import csv
import pickle

from helpers import BaseModel

"""
our 10km X 10km world can be divided into small grids of squares or rectangles
and then be considred their own small worlds on which models were created.


grid defn: x, y, xd, yd
and assume 2*xd < x, 2*yd < y.. don't know if this is necessary.. but for now
im lazy to evaluate

x is the length wise and y is the height wise grid length. xd and yd are the
buffers added on both sides of x and y.

           ....xxxxxxxxxxxxxxxxxxxx....
           .                          .
           y   xxxxxxxxxxxxxxxxxxxx   y
           y   y                  y   y
        y  y   y                  y   y
           y   xxxxxxxxxxxxxxxxxxxx   y
       yd  .                         .
           ....xxxxxxxxxxxxxxxxxxxx....
            xd          x

every individual world has all the training data inside the outer rectangle that includes
the buffer.

when a test data point is considered, you select the world in which the test point falls in
the inner rectangle


TODO:
*) generate training data for diff grids
*) compute grid wise cardinality of each place thats in the grid
*) some mechanism to identify differnet grids
*) given a point in the world identify which grid it falls into
"""


def get_grids(c, X, Y, xd, yd, train = False):
    """
    c: (x, y)
    x: x co-ordinate 3.2314 in kms
    y: y co-ordinate 6.2149 in kms
    X: grid length   800 in mts
    Y: grid height   400 in mts
    xd: x wise buffer 200 in mts
    yd: y wise buffer 100 in mts
    train: boolean to decide if buffer area is considered or not
    """
    grids = []
    if (c[0] < 0) or (c[1] < 0):
        return grids
    if (c[0] > 10) or (c[1] > 10):
        return grids
    grids.append((int(c[0]*1000) / X, int(c[1]*1000) / Y))
    if train == False:
        return grids
    else:
        buffer_points = get_buffer_points(c, xd, yd)
        buffer_grids = map(lambda tc: get_grids(tc, X, Y, xd, yd)[0], buffer_points)
        grids.extend(buffer_grids)
        return list(set(grids))

def get_buffer_points(c, xd, yd):
    points = []
    xdd = xd * 0.001
    ydd = yd * 0.001
    points.extend(map(lambda tx: (tx, c[1]), [c[0] + xdd, c[0] - xdd]))
    points.extend(map(lambda tx: (tx, c[1] + ydd), [c[0], c[0] + xdd, c[0] - xdd]))
    points.extend(map(lambda tx: (tx, c[1] - ydd), [c[0], c[0] + xdd, c[0] - xdd]))
    points = filter(lambda a: a[0] >= 0 and a[1] >= 0 and a[0] <= 10 and a[1] <= 10, points)
    return points

def increment(s, key):
    if key in s:
        s[key] += 1
    else:
        s[key] = 1

def suffix(m, n):
    return '_' + str(m) + '_' + str(n) + '.csv'

def generate_grid_wise_cardinality_and_training_files(train_file, X, Y, xd, yd, pref= 'test'):
    m, n = get_grids((10.0000, 10.0000), X, Y, xd, yd)[0]
    cardinality = [[{} for t in range(n + 1)] for s in range(m + 1)]
    folder_name = '../' + '_'.join([pref, str(X), str(Y), str(xd), str(yd)]) + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    file_handles = [[open(folder_name + 'grid_data' + suffix(s, t), 'wb')\
        for t in range(n + 1)]\
        for s in range(m + 1)]
    cardinality_pickle = folder_name + 'cardinality_pickle.pkl'
    status = folder_name + 'status.pkl'
    f = file(train_file, 'rb')
    fcsv = csv.reader(f)
    progress = 0
    while True:
        try:
            a = fcsv.next() 
            line = ','.join(a) + '\n'
            c = (float(a[1]), float(a[2]))
            place_id = int(a[-1])
            grids = get_grids(c, X, Y, xd, yd, train = True)
            for grid in grids:
                file_handles[grid[0]][grid[1]].write(line)
                increment(cardinality[grid[0]][grid[1]], place_id)
            progress += 1
            if progress % 1000000 == 0:
                print "parsing line %s" %(progress)
        except Exception, e:
            print e
            print a
            break
    temp = [map(lambda file_handle: file_handle.close(), row) for row in file_handles]
    pickle.dump(cardinality, open(cardinality_pickle, 'wb'))
    pickle.dump(True, open(status, 'wb'))
    pass

def get_top_3_places_of_dict(cardinality_dict):
    """
    given the cardinality set of a single grid get the top three places of that grid
    """
    sorted_cardinality = sorted(cardinality_dict.items(), cmp = lambda x, y: cmp(x[1], y[1]),\
        reverse = True)
    return map(lambda x: x[0], sorted_cardinality[:3])

class top_3_grid_places_model(BaseModel):

    def __init__(self, cv_file = '../main_cv_0.02_5.csv', \
        test_file = '../test.csv', X = 800, Y = 400, xd = 200, yd = 100, pref = 'grid'):

        self.cross_validation_file = cv_file
        self.test_file = test_file
        self.X = X
        self.Y = Y
        self.xd = xd
        self.yd = yd
        self.pref = pref

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
        c = (float(row[0]), float(row[1]))
        m, n = get_grids(c, self.X, self.Y, self.xd, self.yd)[0]
        return map(str, self.M[m][n])
