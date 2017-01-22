'''
Created on Dec, 2016

@author: hugo

'''
import argparse

from autoencoder.datasets.wiki10plus import construct_train_test_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus dir')
    parser.add_argument('-o', '--output', type=str, default='./', help='path to the output dir')
    parser.add_argument('-ts', '--test_split', type=float, required=True, help='fraction of the dataset to be used as test data')
    parser.add_argument('-vs', '--vocab_size', type=int, default=2000, help='vocabulary (default 2000)')
    parser.add_argument('-vt', '--vocab_threshold', type=int, default=10, help='vocabulary threshold (default 10), disabled when vocab_size is set')
    args = parser.parse_args()

    construct_train_test_corpus(args.input, args.test_split, args.output, threshold=args.vocab_threshold, topn=args.vocab_size)

if __name__ == '__main__':
    main()
