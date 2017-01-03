'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import sys
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus, corpus2libsvm


def main():
    usage = 'python doc_nade.py [train_path] [test_path] [out_path]'
    try:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_path = sys.argv[3]
    except:
        print usage
        sys.exit()

    docs = load_corpus(train_path)['docs'].items()
    test_docs = load_corpus(test_path)['docs']

    np.random.seed(0)
    np.random.shuffle(docs)
    n_docs = len(docs)
    train_ratio = .9
    train_docs = dict(docs[:int(train_ratio * n_docs)])
    val_docs = dict(docs[int(train_ratio * n_docs):])

    train = corpus2libsvm(train_docs, os.path.join(out_path, 'train.libsvm'))
    val = corpus2libsvm(val_docs, os.path.join(out_path, 'val.libsvm'))
    test = corpus2libsvm(test_docs, os.path.join(out_path, 'test.libsvm'))
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
