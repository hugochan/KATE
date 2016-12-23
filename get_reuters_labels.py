'''
Created on Dec, 2016

@author: hugo

'''
import argparse

from autoencoder.datasets.reuters import extract_labels
from autoencoder.utils.io_utils import load_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--label', type=str, required=True, help='path to the input label file')
    parser.add_argument('-c', '--corpus', type=str, required=True, help='path to the constructed corpus file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to the output file')
    args = parser.parse_args()

    docs_names = load_json(args.corpus)['docs'].keys()
    extract_labels(docs_names, args.label, args.output)
