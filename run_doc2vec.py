'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus
from autoencoder.baseline.doc2vec import doc2vec
from autoencoder.utils.io_utils import dump_json, write_file
from autoencoder.utils.op_utils import revdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, type=str, help='path to the corpus file')
    parser.add_argument('-mf', '--mod_file', required=True, type=str, help='path to the word2vec mod file')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    docs, vocab_dict = corpus['docs'], corpus['vocab']

    doc_codes = doc2vec(docs, revdict(vocab_dict), args.mod_file, args.output, size=300)
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
