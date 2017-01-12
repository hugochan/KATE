'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder

from autoencoder.testing.classifier import neural_classifier
from autoencoder.utils.io_utils import load_json, load_marshal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc codes file')
    parser.add_argument('train_doc_labels', type=str, help='path to the train doc labels file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    args = parser.parse_args()

    # autoencoder
    train_doc_codes = load_json(args.train_doc_codes)
    train_doc_labels = load_json(args.train_doc_labels)
    test_doc_codes = load_json(args.test_doc_codes)
    test_doc_labels = load_json(args.test_doc_labels)
    X_train = np.r_[train_doc_codes.values()]
    Y_train = [train_doc_labels[i] for i in train_doc_codes.keys()]
    X_test = np.r_[test_doc_codes.values()]
    Y_test = [test_doc_labels[i] for i in test_doc_codes.keys()]

    # # DBN
    # X_train = np.array(load_marshal(args.train_doc_codes))
    # Y_train = load_marshal(args.train_doc_labels)
    # X_test = np.array(load_marshal(args.test_doc_codes))
    # Y_test = load_marshal(args.test_doc_labels)

    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train = np_utils.to_categorical(encoder.transform(Y_train))
    Y_test = np_utils.to_categorical(encoder.transform(Y_test))

    seed = 7
    np.random.seed(seed)
    val_idx = np.random.choice(range(X_train.shape[0]), args.n_val, replace=False)
    train_idx = list(set(range(X_train.shape[0])) - set(val_idx))
    X_new_train = X_train[train_idx]
    Y_new_train = Y_train[train_idx]
    X_new_val = X_train[val_idx]
    Y_new_val = Y_train[val_idx]
    print 'train: %s, val: %s, test: %s' % (X_new_train.shape[0], X_new_val.shape[0], X_test.shape[0])
    results = neural_classifier(X_new_train, Y_new_train, X_new_val, Y_new_val, \
            X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed)
    print 'acc on test set: %s' % results
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
