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

def generate_grid_wise_cardinality_and_training_files(train_file, X, Y, xd, yd,\
    pref= 'test', files = False):
    m, n = get_grids((10.0000, 10.0000), X, Y, xd, yd)[0]
    cardinality = [[{} for t in range(n + 1)] for s in range(m + 1)]
    folder_name = '../' + '_'.join([pref, str(X), str(Y), str(xd), str(yd)]) + '/'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    if files:
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
                if files:
                    file_handles[grid[0]][grid[1]].write(line)
                increment(cardinality[grid[0]][grid[1]], place_id)
            progress += 1
            if progress % 1000000 == 0:
                print "parsing line %s" %(progress)
        except Exception, e:
            print e
            print a
            break
    if files:
        temp = [map(lambda file_handle: file_handle.close(), row) for row in file_handles]
    pickle.dump(cardinality, open(cardinality_pickle, 'wb'))
    pickle.dump(True, open(status, 'wb'))

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
        self.description = 'top_3_grid_places_model_' + str(X) + '_' + \
            str(Y) + '_' + \
            str(xd) + '_' + \
            str(yd) + '_'
        print "@@@@@@@@@@@@@@@@@@@"
        print self.description

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

class Grid(object):
    def __init__(self, X = 800, Y = 400, xd = 200, yd = 100,\
        pref = 'test', files_flag = False, train_file = '../main_train_0.02_5.csv'):
        """
        X, Y, xd, yd :: grid definitions
        pref: prefix with which the folder that contains the cardinality matrix
            and etc stays
            if pref == test, it considers the test file '../code_train.csv'
            else it considers the train file '../main_train_0.02_5.csv'
        files_flag: specifies if individual grid files are to be generated along with
            the cardinality matrix
        """
        self.X = X
        self.Y = Y
        self.xd = xd
        self.yd = yd
        self.pref = pref
        self.files_flag = files_flag
        self.train_file = train_file
        if self.pref == 'test':
            self.train_file = '../code_train.csv'
        else:
            self.train_file = self.train_file
        self.max_m, self.max_n = get_grids((10.0000, 10.0000), X, Y, xd, yd)[0]

    def getFolder(self):
        return '../' + '_'.join([self.pref, str(self.X), str(self.Y), str(self.xd), str(self.yd)]) + '/'

    def getGirdFiles(self):
        c = itertools.product(range(self.max_m + 1), range(self.max_ni + 1))
        grid_files = map(lambda x: self.getFolder() + '_'.join(['grid_data', str(x[0]), str(x[1])])\
            + '.csv', c)
        return list(grid_files)

    def getParamsFile(self, rx, ry):
        rx = 'rx' + str(rx)
        ry = 'ry' + str(ry)
        paramsFile = self.getFolder() + '_'.join(['grid', str(self.X), str(self.Y), str(self.xd), str(self.yd),\
            rx, ry, 'params_dict.pickle'])
        if os.path.exists(paramsFile):
            return paramsFile
        else:
            return None

    def getGridFile(self, m, n):
        return self.getFolder() + '_'.join(['grid_data', str(m), str(n)]) + '.csv'

    def generateCardinalityMatrix(self):
        folder_name = self.getFolder()
        status = folder_name + 'status.pkl'
        cardinality = folder_name + 'cardinality_pickle.pkl'
        if not os.path.exists(status):
            print "Generating grid cardinality matrix of grid"
            generate_grid_wise_cardinality_and_training_files(self.train_file,\
                self.X, self.Y, self.xd, self.yd, self.pref, self.files_flag)
        self.M = pickle.load(open(cardinality, 'rb'))

if __name__ == '__main__':
    f = open('../cv_results_run_5.txt', 'ab')
    # Xs = range(10, 30, 4)
    # Ys = range(10, 30, 4)
    # xd = 4
    # yd = 1
    # for X in Xs:
    #     for Y in Ys:
    #         if X > Y:
    #             model = top_3_grid_places_model(X = X, Y = Y, xd = xd, yd = yd)
    #             cv = model.get_cross_validation_mean_precision()
    #             line = model.description + ':::' + str(cv) + '\n'
    #             f.write(line)
    model = top_3_grid_places_model(X = 10, Y = 5, xd = 4, yd = 1)
    cv = model.get_cross_validation_mean_precision()
    line = model.description + ':::' + str(cv) + '\n'
    f.write(line)
    f.close()
