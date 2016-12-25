'''
Created on Dec, 2016

@author: hugo

'''

import os
import sys
from autoencoder.preprocessing.preprocessing import construct_train_test_corpus, generate_20news_doc_labels


def main():
    usage = 'python construct_corpus.py [train_path] [test_path] [out_path]'
    try:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_path = sys.argv[3]
    except:
        print usage
        sys.exit()

    train_corpus, test_corpus = construct_train_test_corpus(train_path, test_path, out_path, threshold=5, topn=2000)
    train_labels = generate_20news_doc_labels(train_corpus['docs'].keys(), os.path.join(out_path, 'train.labels'))
    test_labels = generate_20news_doc_labels(test_corpus['docs'].keys(), os.path.join(out_path, 'test.labels'))

if __name__ == "__main__":
    main()
