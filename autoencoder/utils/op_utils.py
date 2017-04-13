'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import numpy as np


def calc_ranks(x):
    """Given a list of items, return a list(in ndarray type) of ranks.
    """
    n = len(x)
    index = list(zip(*sorted(list(enumerate(x)), key=lambda d:d[1], reverse=True))[0])
    rank = np.zeros(n)
    rank[index] = range(1, n + 1)
    return rank

def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

def l1norm(x):
    return x / sum([np.abs(y) for y in x])

def vecnorm(vec, norm, epsilon=1e-3):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    """
    if norm not in ('prob', 'max1', 'logmax1'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms include 'prob',\
             'max1' and 'logmax1'." % norm)

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'prob':
            veclen = np.sum(np.abs(vec)) + epsilon * len(vec) # smoothing
        elif norm == 'max1':
            veclen = np.max(vec) + epsilon
        elif norm == 'logmax1':
            vec = np.log10(1. + vec)
            veclen = np.max(vec) + epsilon
        if veclen > 0.0:
            return (vec + epsilon) / veclen
        else:
            return vec
    else:
        raise ValueError('vec should be ndarray, found: %s' % type(vec))

def unitmatrix(matrix, norm='l2', axis=1):
    if norm == 'l1':
        maxtrixlen = np.sum(np.abs(matrix), axis=axis)
    if norm == 'l2':
        maxtrixlen = np.linalg.norm(matrix, axis=axis)

    if np.any(maxtrixlen <= 0):
        return matrix
    else:
        maxtrixlen = maxtrixlen.reshape(1, len(maxtrixlen)) if axis == 0 else maxtrixlen.reshape(len(maxtrixlen), 1)
        return matrix / maxtrixlen

def add_gaussian_noise(X, corruption_ratio, range_=[0, 1]):
    X_noisy = X + corruption_ratio * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = np.clip(X_noisy, range_[0], range_[1])

    return X_noisy

def add_masking_noise(X, fraction):
    assert fraction >= 0 and fraction <= 1
    X_noisy = np.copy(X)
    nrow, ncol = X.shape
    n = int(ncol * fraction)
    for i in range(nrow):
        idx_noisy  = np.random.choice(ncol, n, replace=False)
        X_noisy[i, idx_noisy] = 0

    return X_noisy

def add_salt_pepper_noise(X, fraction):
    assert fraction >= 0 and fraction <= 1
    X_noisy = np.copy(X)
    nrow, ncol = X.shape
    n = int(ncol * fraction)
    for i in range(nrow):
        idx_noisy  = np.random.choice(ncol, n, replace=False)
        X_noisy[i, idx_noisy] = np.random.binomial(1, .5, n)

    return X_noisy

