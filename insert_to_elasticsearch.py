

from elasticsearch import Elasticsearch
import csv
import time
import datetime
from itertools import imap
from elasticsearch.helpers import bulk




es = Elasticsearch('127.0.0.1')

def parse_with_timestamp_train(row, t):
    return [int(row[0]), float(row[1]), float(row[2]), int(row[3]), float(row[4]) + t, int(row[5])]

def get_csv_iterator(file_name):
    f = open(file_name, 'rb')
    fcsv = csv.reader(f)
    return fcsv

def bulk_operation_train(row, index_name, type_name):
    return {'x': float(row[1]),
           'y': float(row[2]),
           'accuracy': int(row[3]),
           'time': datetime.datetime(2016, 5, 1) + datetime.timedelta(seconds = int(row[4])),
           'place_id': int(row[5]),
           '_op_type': 'create',
           '_index': index_name,
           '_type': type_name,
           '_id': row[0],
            }

def bulk_operation_test(row, index_name, type_name):
    return {'x': float(row[1]),
           'y': float(row[2]),
           'accuracy': int(row[3]),
           'time': datetime.datetime(2016, 5, 1) + datetime.timedelta(seconds = int(row[4])),
           '_op_type': 'create',
           '_index': index_name,
           '_type': type_name,
           '_id': row[0],
            }


def get_bulk_action_iterator(file_name, index, train = True):
    if train:
        return imap(lambda x: bulk_operation_train(x, index, 'event'), get_csv_iterator(file_name))
    else:
        return imap(lambda x: bulk_operation_test(x, index, 'event'), get_csv_iterator(file_name))

if __name__ == '__main__':
    # bulk(es, get_bulk_action_iterator('code_train.csv', 'code_train'))
    # bulk(es, get_bulk_action_iterator('train.csv', 'train'))
    bulk(es, get_bulk_action_iterator('test.csv', 'test', train = False))