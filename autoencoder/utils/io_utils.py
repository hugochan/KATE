'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

import json
import cPickle as pickle
import marshal as m


def dump_marshal(data, path_to_file):
    try:
        with open(path_to_file, 'w') as f:
            m.dump(data, f)
    except Exception as e:
        raise e

def load_marshal(path_to_file):
    try:
        with open(path_to_file, 'r') as f:
            data = m.load(f)
    except Exception as e:
        raise e

    return data

def dump_pickle(data, path_to_file):
    try:
        with open(path_to_file, 'w') as f:
            pickle.dump(data, f)
    except Exception as e:
        raise e

def load_pickle(path_to_file):
    try:
        with open(path_to_file, 'r') as f:
            data = pickle.load(f)
    except Exception as e:
        raise e

    return data

def dump_json(data, file):
    try:
        with open(file, 'w') as datafile:
            json.dump(data, datafile)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as datafile:
            data = json.load(datafile)
    except Exception as e:
        raise e

    return data

def write_file(data, file):
    try:
        with open(file, 'w') as datafile:
            for line in data:
                datafile.write(' '.join(line) + '\n')
    except Exception as e:
        raise e

def load_file(file, float_=False):
    data = []
    try:
        with open(file, 'r') as datafile:
            for line in datafile:
                content = line.strip('\n').split()
                if float_:
                    content = [float(x) for x in content]
                data.append(content)
    except Exception as e:
        raise e

    return data
