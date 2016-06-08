

from helpers import compute_place_wise_stats

def compute_train():
    for i in range(1,23):
        inp = '../split_train/' + str(i) + '_train00'
        out = '../split_train/' + str(i) + 'place'
        compute_place_wise_stats(inp, out)

def test_compute_train():
    compute_place_wise_stats('../code_train.csv', '../code_train_stats.csv')


if __name__ == '__main__':
    test_compute_train()
