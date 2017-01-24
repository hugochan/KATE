'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import numpy as np
from collections import defaultdict, Counter
from keras.models import Sequential
from keras.layers import Dense
# from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from autoencoder.utils.op_utils import unitmatrix


def retrieval(X_train, Y_train, X_test, Y_test, fractions=[0.01, 0.5, 1.0], multilabel=False):
    db_size = len(X_train)
    n_queries = len(X_test)
    X_train = unitmatrix(X_train) # normalize
    X_test = unitmatrix(X_test)
    score = X_test.dot(X_train.T)
    X_train = None
    X_test = None
    precisions = defaultdict(float)

    for idx in range(n_queries):
        retrieval_idx = score[idx].argsort()[::-1]
        target = Y_test[idx]
        for fr in fractions:
            ntop = int(fr * db_size)
            pr = float(len([i for i in retrieval_idx[:ntop] if hit(Y_train[i], target, multilabel)])) / ntop
            precisions[fr] += pr
    precisions = dict([(x, y / n_queries) for x, y in precisions.iteritems()])

    return sorted(precisions.items(), key=lambda d:d[0])

def retrieval_by_doclength(X_train, Y_train, X_test, Y_test, len_test, fraction=0.001, len_bin=600, multilabel=False):
    X_train = unitmatrix(X_train) # normalize
    X_test = unitmatrix(X_test)
    score = X_test.dot(X_train.T)
    precisions = defaultdict(list)
    n_queries = len(X_test)
    ntop = int(fraction * len(X_train))
    # bins = [50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]
    bins = [100, 120, 150, 200, 300, 1000, 1500, 2000, 4000]

    for idx in range(n_queries):
        retrieval_idx = score[idx].argsort()[::-1]
        pr = float(len([i for i in retrieval_idx[:ntop] if hit(Y_train[i], Y_test[idx], multilabel)])) / ntop
        for each in bins:
            if len_test[idx] < each:
                precisions[each].append(pr)
                break
    import pdb;pdb.set_trace()
    precisions = dict([(x, sum(y) / len(y)) for x, y in precisions.iteritems()])

    return sorted(precisions.items(), key=lambda d:d[0])


def retrieval_perlabel(X_train, Y_train, X_test, Y_test, fractions=[0.01, 0.5, 1.0]):
    X_train = unitmatrix(X_train) # normalize
    X_test = unitmatrix(X_test)
    score = X_test.dot(X_train.T)
    precisions = defaultdict(dict)
    label_counter = Counter(Y_test.tolist())

    for idx in range(len(X_test)):
        retrieval_idx = score[idx].argsort()[::-1]
        for fr in fractions:
            ntop = int(fr * len(X_train))
            pr = float(len([i for i in retrieval_idx[:ntop] if Y_train[i] == Y_test[idx]])) / ntop
            try:
                precisions[fr][Y_test[idx]] += pr
            except:
                precisions[fr][Y_test[idx]] = pr
    new_pr = {}
    for fr, val in precisions.iteritems():
        avg_pr = 0.
        for label, pr in val.iteritems():
            avg_pr += pr / label_counter[label]
        new_pr[fr] = avg_pr / len(label_counter)

    return sorted(new_pr.items(), key=lambda d:d[0])

def hit(x, y, multilabel=False):
    if multilabel:
        return len(set(x).intersection(set(y))) > 0
    else:
        return x == y


