'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np

from autoencoder.testing.classifier import classifier
from autoencoder.utils.io_utils import load_json, load_marshal

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc codes file')
    parser.add_argument('train_doc_labels', type=str, help='path to the train doc labels file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
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

    results = classifier(X_train, Y_train, X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size)
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
