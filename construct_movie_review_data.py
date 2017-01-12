'''
Created on Jan, 2017

@author: hugo

'''
import argparse

from autoencoder.datasets.movie_review_data import construct_train_test_corpus


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-o', '--output', type=str, default='./', help='path to the output dir')
    parser.add_argument('-ts', '--test_split', type=float, required=True, help='fraction of the dataset to be used as test data')
    args = parser.parse_args()

    construct_train_test_corpus(args.input, args.test_split, args.output, threshold=10, topn=2000)

if __name__ == '__main__':
    main()
