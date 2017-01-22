'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus, corpus2libsvm
from autoencoder.utils.io_utils import load_file, dump_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_doc_codes', type=str, help='path to the train doc code file')
    parser.add_argument('train_doc_names', type=str, help='path to the train doc name file')
    parser.add_argument('val_doc_codes', type=str, help='path to the valid doc code file')
    parser.add_argument('val_doc_names', type=str, help='path to the valid doc name file')
    parser.add_argument('test_doc_codes', type=str, help='path to the test doc code file')
    parser.add_argument('test_doc_names', type=str, help='path to the test doc name file')
    parser.add_argument('out_dir', type=str, help='path to the output dir')
    args = parser.parse_args()

    train_doc_codes_path = args.train_doc_codes
    test_doc_codes_path = args.test_doc_codes
    train_doc_codes = load_file(train_doc_codes_path, True)
    train_doc_names = load_file(args.train_doc_names)
    val_doc_codes = load_file(args.val_doc_codes, True)
    val_doc_names = load_file(args.val_doc_names)
    test_doc_codes = load_file(test_doc_codes_path, True)
    test_doc_names = load_file(args.test_doc_names)

    assert len(train_doc_codes) == len(train_doc_names)
    assert len(val_doc_codes) == len(val_doc_names)
    assert len(test_doc_codes) == len(test_doc_names)

    new_train_doc_codes = {}
    new_test_doc_codes = {}

    for i in range(len(train_doc_names)):
        new_train_doc_codes[''.join(train_doc_names[i])] = train_doc_codes[i]
    del train_doc_codes
    for i in range(len(val_doc_names)):
        new_train_doc_codes[''.join(val_doc_names[i])] = val_doc_codes[i]
    del val_doc_codes
    for i in range(len(test_doc_names)):
        new_test_doc_codes[''.join(test_doc_names[i])] = test_doc_codes[i]
    del test_doc_codes

    out_dir = args.out_dir
    dump_json(new_train_doc_codes, os.path.join(out_dir, 'new_' + os.path.basename(train_doc_codes_path)))
    dump_json(new_test_doc_codes, os.path.join(out_dir, 'new_' + os.path.basename(test_doc_codes_path)))

    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
