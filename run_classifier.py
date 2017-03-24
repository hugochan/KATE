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
    parser.add_argument('train_doc_labels', type=str, help='path to the train doc labels file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc codes file')
    parser.add_argument('test_doc_labels', type=str, help='path to the test doc labels file')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-cv', '--cross_validation', type=int, help='k-fold cross validation')
    parser.add_argument('-mlc', '--multilabel_clf', action='store_true', help='multilabel classification flag')

    args = parser.parse_args()

    # autoencoder
    train_doc_codes = load_json(args.train_doc_codes)
    train_doc_labels = load_json(args.train_doc_labels)
    test_doc_codes = load_json(args.test_doc_codes)
    test_doc_labels = load_json(args.test_doc_labels)
    X_train = np.r_[train_doc_codes.values()]
    Y_train = [train_doc_labels[i] for i in train_doc_codes]
    X_test = np.r_[test_doc_codes.values()]
    Y_test = [test_doc_labels[i] for i in test_doc_codes]

    # # DBN
    # X_train = np.array(load_pickle(args.train_doc_codes))
    # Y_train = load_pickle(args.train_doc_labels)
    # X_test = np.array(load_pickle(args.test_doc_codes))
    # Y_test = load_pickle(args.test_doc_labels)
    # import pdb;pdb.set_trace()

    if args.multilabel_clf:
        encoder = MultiLabelBinarizer()
        encoder.fit(Y_train + Y_test)
        Y_train = encoder.transform(Y_train)
        Y_test = encoder.transform(Y_test)
    else:
        Y = Y_train + Y_test
        n_train = len(Y_train)
        n_test = len(Y_test)
        encoder = LabelEncoder()
        Y = np_utils.to_categorical(encoder.fit_transform(Y))
        Y_train = Y[:n_train]
        Y_test = Y[-n_test:]

    seed = 7
    np.random.seed(seed)
    if not args.cross_validation:
        val_idx = np.random.choice(range(X_train.shape[0]), args.n_val, replace=False)
        train_idx = list(set(range(X_train.shape[0])) - set(val_idx))
        X_new_train = X_train[train_idx]
        Y_new_train = Y_train[train_idx]
        X_new_val = X_train[val_idx]
        Y_new_val = Y_train[val_idx]
        print 'train: %s, val: %s, test: %s' % (X_new_train.shape[0], X_new_val.shape[0], X_test.shape[0])
        if args.multilabel_clf:
            results = multilabel_classifier(X_new_train, Y_new_train, X_new_val, Y_new_val, \
                    X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed)
            print 'f1 score on test set: macro_f1: %s, micro_f1: %s' % tuple(results)
        else:
            results = multiclass_classifier(X_new_train, Y_new_train, X_new_val, Y_new_val, \
                    X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed)
            print 'acc on test set: %s' % results
    else:
        X = np.concatenate((X_train, X_test), axis=0)
        Y = np.concatenate((Y_train, Y_test), axis=0)
        ss = ShuffleSplit(n_splits=int(args.cross_validation), test_size=X_test.shape[0], random_state=seed)
        results = []
        for train_idx, test_idx in ss.split(X):
            val_idx = np.random.choice(train_idx, args.n_val, replace=False)
            new_train_idx = list(set(train_idx) - set(val_idx))
            X_new_train = X[new_train_idx]
            Y_new_train = Y[new_train_idx]
            X_new_val = X[val_idx]
            Y_new_val = Y[val_idx]
            if args.multilabel_clf:
                results.append(multilabel_classifier(X_new_train, Y_new_train, X_new_val, Y_new_val, \
                        X_test, Y_test, nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed))
            else:
                results.append(multiclass_classifier(X_new_train, Y_new_train, X_new_val, Y_new_val, \
                    X[test_idx], Y[test_idx], nb_epoch=args.n_epoch, batch_size=args.batch_size, seed=seed))

        if args.multilabel_clf:
            macro_f1, micro_f1 = zip(*results)
            macro_mean = np.mean(macro_f1)
            macro_std = np.std(macro_f1)
            micro_mean = np.mean(micro_f1)
            micro_std = np.std(micro_f1)
            print 'f1 score on %s-fold cross validation: macro_f1: %s (%s), micro_f1: %s (%s)' \
                    % (int(args.cross_validation), macro_mean, macro_std, micro_mean, micro_std)
        else:
            mean = np.mean(results)
            std = np.std(results)
            print 'acc on %s-fold cross validation: %s (%s)' % (int(args.cross_validation), mean, std)
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
