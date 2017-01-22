'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus, corpus2libsvm
from autoencoder.utils.io_utils import load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str, help='path to the train corpus file')
    parser.add_argument('test_path', type=str, help='path to the test corpus file')
    parser.add_argument('train_label', type=str, help='path to the train label file')
    parser.add_argument('test_label', type=str, help='path to the test label file')
    parser.add_argument('out_dir', type=str, help='path to the output dir')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='validation set size')
    args = parser.parse_args()

    docs = load_corpus(args.train_path)['docs'].items()
    test_docs = load_corpus(args.test_path)['docs']

    np.random.seed(0)
    np.random.shuffle(docs)
    n_val = args.n_val
    train_docs = dict(docs[:-n_val])
    val_docs = dict(docs[-n_val:])

    # doc_labels = load_json(args.train_label)
    # test_labels = load_json(args.test_label)
    doc_labels = None
    test_labels = None
    train = corpus2libsvm(train_docs, doc_labels, os.path.join(args.out_dir, 'train.libsvm'))
    val = corpus2libsvm(val_docs, doc_labels, os.path.join(args.out_dir, 'val.libsvm'))
    test = corpus2libsvm(test_docs, test_labels, os.path.join(args.out_dir, 'test.libsvm'))

    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
