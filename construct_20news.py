'''
Created on Dec, 2016

@author: hugo

'''

import os
import argparse

from autoencoder.preprocessing.preprocessing import construct_train_test_corpus, generate_20news_doc_labels#, generate_8k_doc_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train', '--train_path', type=str, required=True, help='path to the training corpus')
    parser.add_argument('-test', '--test_path', type=str, required=True, help='path to the test corpus')
    parser.add_argument('-o', '--out_dir', type=str, required=True, help='path to the output dir')
    parser.add_argument('-threshold', '--threshold', type=int, default=5, help='word frequency threshold (default 5)')
    parser.add_argument('-topn', '--topn', type=int, default=2000, help='top n words (default 2000)')
    args = parser.parse_args()

    train_corpus, test_corpus = construct_train_test_corpus(args.train_path, args.test_path, args.out_dir, threshold=args.threshold, topn=args.topn)
    train_labels = generate_20news_doc_labels(train_corpus['docs'].keys(), os.path.join(args.out_dir, 'train.labels'))
    test_labels = generate_20news_doc_labels(test_corpus['docs'].keys(), os.path.join(args.out_dir, 'test.labels'))
    # train_labels = generate_8k_doc_labels(train_corpus['docs'].keys(), os.path.join(args.out_dir, 'train.labels'))
    # test_labels = generate_8k_doc_labels(test_corpus['docs'].keys(), os.path.join(args.out_dir, 'test.labels'))

if __name__ == "__main__":
    main()
