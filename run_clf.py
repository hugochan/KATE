'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import ShuffleSplit

from autoencoder.testing.classifier import multiclass_classifier, multilabel_classifier
from autoencoder.utils.io_utils import load_json, load_pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc codes file')
    parser.add_argument('train_doc_labels', type=str, help='path to the train doc codes file')
    parser.add_argument('val_doc_codes', type=str, help='path to the train doc codes file')
    parser.add_argument('val_doc_labels', type=str, help='path to the train doc labels file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-mlc', '--multilabel_clf', action='store_true', help='multilabel classification flag')

    args = parser.parse_args()

    # autoencoder
    train_doc_codes = load_json(args.train_doc_codes)
    train_doc_labels = load_json(args.train_doc_labels)
    val_doc_codes = load_json(args.val_doc_codes)
    val_doc_labels = load_json(args.val_doc_labels)
    test_doc_codes = load_json(args.test_doc_codes)
    test_doc_labels = load_json(args.test_doc_labels)
    X_train = np.r_[train_doc_codes.values()]
    Y_train = [train_doc_labels[i] for i in train_doc_codes]
    X_val = np.r_[val_doc_codes.values()]
    Y_val = [val_doc_labels[i] for i in val_doc_codes]
    X_test = np.r_[test_doc_codes.values()]
    Y_test = [test_doc_labels[i] for i in test_doc_codes]

    # # DBN
    # X_train = np.array(load_pickle(args.train_doc_codes))
    # Y_train = load_pickle(args.train_doc_labels)
    # X_val = np.array(load_pickle(args.val_doc_codes))
    # Y_val = load_pickle(args.val_doc_labels)
    # X_test = np.array(load_pickle(args.test_doc_codes))
    # Y_test = load_pickle(args.test_doc_labels)

    if args.multilabel_clf:
        encoder = MultiLabelBinarizer()
        encoder.fit(Y_train + Y_val + Y_test)
        Y_train = encoder.transform(Y_train)
        Y_val = encoder.transform(Y_val)
        Y_test = encoder.transform(Y_test)
    else:
        Y = Y_train + Y_val + Y_test
        n_train = len(Y_train)
        n_val = len(Y_val)
        n_test = len(Y_test)
        encoder = LabelEncoder()
        Y = np_utils.to_categorical(encoder.fit_transform(Y))
        Y_train = Y[:n_train]
        Y_val = Y[n_train:n_train + n_val]
        Y_test = Y[-n_test:]

    seed = 7
    print 'train: %s, val: %s, test: %s' % (X_train.shape[0], X_val.shape[0], X_test.shape[0])
    if args.multilabel_clf:
        results = multilabel_classifier(X_train, Y_train, X_val, Y_val, \
                X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed)
        print 'f1 score on test set: macro_f1: %s, micro_f1: %s' % tuple(results)
    else:
        results = multiclass_classifier(X_train, Y_train, X_val, Y_val, \
                X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed)
        print 'acc on test set: %s' % results

    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
