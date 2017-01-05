'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np

from autoencoder.utils.io_utils import load_json, dump_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc code file')
    parser.add_argument('val_doc_codes', type=str, help='path to the valid doc code file')
    parser.add_argument('out_dir', type=str, help='path to the output dir')
    args = parser.parse_args()

    train_doc_codes_path = args.train_doc_codes
    train_doc_codes = load_json(train_doc_codes_path)
    val_doc_codes = load_json(args.val_doc_codes)

    # import pdb;pdb.set_trace()

    train_doc_codes.update(val_doc_codes)


    out_dir = args.out_dir
    dump_json(train_doc_codes, os.path.join(out_dir, 'new_' + os.path.basename(train_doc_codes_path)))


    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
