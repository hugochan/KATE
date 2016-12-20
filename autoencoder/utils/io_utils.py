'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

import json
import cPickle as pickle


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
