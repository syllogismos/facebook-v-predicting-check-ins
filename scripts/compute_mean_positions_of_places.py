

from helpers import compute_place_wise_stats, compute_mode_xy

def compute_train():
    for i in range(1,23):
        inp = '../split_train/' + str(i) + '_train00'
        out = '../split_train/' + str(i) + '_place'
        compute_place_wise_stats(inp, out)

def test_compute_train():
    compute_place_wise_stats('../code_train.csv', '../code_train_stats.csv')

def compute_modes():
    for i in range(1, 23):
        print "Computing modes for file %s" %(i)
        inp = '../split_train/' + str(i) + '_train00'
        out = '../split_train/' + str(i) + '_mode'
        compute_mode_xy(inp, out)

def test_compute_mode():
    compute_mode_xy('../code_train.csv', '../code_train_mode.csv')



if __name__ == '__main__':
    compute_modes()
