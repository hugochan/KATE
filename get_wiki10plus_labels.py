'''
Created on Dec, 2016

@author: hugo

'''
import argparse

from autoencoder.datasets.wiki10plus import extract_labels
from autoencoder.utils.io_utils import load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True, help='path to the input label file')
    parser.add_argument('-c', '--corpus', type=str, required=True, help='path to the constructed corpus file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to the output file')
    args = parser.parse_args()


    extract_labels(load_json(args.corpus)['docs'], load_json(args.label), args.output)


if __name__ == '__main__':
    main()
