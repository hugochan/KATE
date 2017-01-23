'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np
from keras.utils import np_utils

from autoencoder.testing.retrieval import retrieval, retrieval_by_doclength
from autoencoder.utils.io_utils import load_json, load_marshal
from autoencoder.preprocessing.preprocessing import load_corpus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc codes file')
    parser.add_argument('train_doc_labels', type=str, help='path to the train doc labels file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-qi', '--query_info', type=str, help='path to the query corpus (for geting doc length info)')
    parser.add_argument('-ml', '--multilabel', action='store_true', help='multilabel flag')
    args = parser.parse_args()


    # autoencoder
    train_doc_codes = load_json(args.train_doc_codes)
    train_doc_labels = load_json(args.train_doc_labels)
    test_doc_codes = load_json(args.test_doc_codes)
    test_doc_labels = load_json(args.test_doc_labels)
    X_train = np.r_[train_doc_codes.values()]
    Y_train = np.array([train_doc_labels[i] for i in train_doc_codes])
    X_test = np.r_[test_doc_codes.values()]
    Y_test = np.array([test_doc_labels[i] for i in test_doc_codes])

    # # DocNADE
    # train_doc_codes = load_json(args.train_doc_codes)
    # train_doc_labels = load_json(args.train_doc_labels)
    # test_doc_codes = load_json(args.test_doc_codes)
    # test_doc_labels = load_json(args.test_doc_labels)
    # X_train = []
    # for each in train_doc_codes.values():
    #     X_train.append([float(x) for x in each])
    # X_test = []
    # for each in test_doc_codes.values():
    #     X_test.append([float(x) for x in each])

    # X_train = np.r_[X_train]
    # Y_train = np.array([train_doc_labels[i] for i in train_doc_codes])
    # X_test = np.r_[X_test]
    # Y_test = np.array([test_doc_labels[i] for i in test_doc_codes])


    # # DBN
    # X_train = np.array(load_marshal(args.train_doc_codes))
    # Y_train = np.array(load_marshal(args.train_doc_labels))
    # X_test = np.array(load_marshal(args.test_doc_codes))
    # Y_test = np.array(load_marshal(args.test_doc_labels))


    seed = 7
    np.random.seed(seed)
    val_idx = np.random.choice(range(X_train.shape[0]), args.n_val, replace=False)
    train_idx = list(set(range(X_train.shape[0])) - set(val_idx))
    X_new_train = X_train[train_idx]
    Y_new_train = Y_train[train_idx]
    X_new_val = X_train[val_idx]
    Y_new_val = Y_train[val_idx]
    print 'train: %s, val: %s, test: %s' % (X_new_train.shape[0], X_new_val.shape[0], X_test.shape[0])

    results = retrieval(X_new_train, Y_new_train, X_new_val, Y_new_val,\
                        fractions=[0.001], multilabel=args.multilabel)
    print 'precision on val set: %s' % results

    if not args.query_info:
        results = retrieval(X_train, Y_train, X_test, Y_test,\
                        fractions=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0], multilabel=args.multilabel)
    else:
        query_docs = load_corpus(args.query_info)['docs']
        len_test = [sum(query_docs[i].values()) for i in test_doc_codes]
        results = retrieval_by_doclength(X_train, Y_train, X_test, Y_test, len_test, fraction=0.001, multilabel=args.multilabel)
    print 'precision on test set: %s' % results
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
