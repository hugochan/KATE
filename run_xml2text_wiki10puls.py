'''
Created on Jan, 2017

@author: hugo

'''
import argparse

from autoencoder.datasets.wiki10plus import xml2text
from autoencoder.utils.io_utils import load_json

def main():
    parser T= argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus dir')
    parser.add_argument('-o', '--output', type=str, default='./', help='path to the output dir')
    parser.add_argument('-wl', '--whitelist', type=str, help='path to the whitelist file')
    args = parser.parse_args()

    if args.whitelist:
        white_list = load_json(args.whitelist)
    else:
        white_list = None

    xml2text(args.input, args.output, white_list)

if __name__ == '__main__':
    main()
