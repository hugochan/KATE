'''
Created on Jan, 2017

@author: hugo

'''
import argparse

from autoencoder.datasets.wikitag_extractor import extract_labels
from autoencoder.utils.io_utils import dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input source file')
    parser.add_argument('--topn', type=int, default=25, help='keep only topn most frequent labels')
    parser.add_argument('-o', '--output', type=str, help='path to the output file')
    args = parser.parse_args()


    labeldict = extract_labels(args.input, args.topn)
    dump_json(labeldict, args.output)

if __name__ == '__main__':
    main()
