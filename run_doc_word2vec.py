'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus
from autoencoder.baseline.doc_word2vec import load_w2v, doc_word2vec, get_similar_words
from autoencoder.utils.io_utils import write_file
from autoencoder.utils.op_utils import revdict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', required=True, type=str, help='path to the corpus file')
    parser.add_argument('-mf', '--mod_file', required=True, type=str, help='path to the word2vec mod file')
    parser.add_argument('-sw', '--sample_words', type=str, help='path to the output sample words file')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    docs, vocab_dict = corpus['docs'], corpus['vocab']
    w2v = load_w2v(args.mod_file)

    # doc_codes = doc_word2vec(w2v, docs, revdict(vocab_dict), args.output, avg=True)
    if args.sample_words:
        queries = ['weapon', 'christian', 'compani', 'israel', 'law', 'hockey', 'comput', 'space']
        words = []
        for each in queries:
            words.append(get_similar_words(w2v, each, topn=5))
        write_file(words, args.sample_words)
        print 'Saved sample words file to %s' % args.sample_words


    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
