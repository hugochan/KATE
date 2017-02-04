'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import timeit
import math
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus
from autoencoder.utils.io_utils import dump_json, write_file
from autoencoder.baseline.lda import train_lda, generate_doc_codes, load_model, show_topics, calc_pairwise_cosine

def train(args):
    corpus = load_corpus(args.corpus)
    docs, vocab_dict = corpus['docs'], corpus['vocab']
    doc_bow = []
    for k in docs.keys():
        bows = []
        for idx, count in docs[k].iteritems():
            bows.append((int(idx), count))
        doc_bow.append(bows)
        del docs[k]
    vocab_dict = dict([(int(y), x) for x, y in vocab_dict.iteritems()])

    start = timeit.default_timer()
    train_lda(doc_bow, vocab_dict, args.n_topics, args.n_iter, args.save_model)
    print 'runtime: %ss' % (timeit.default_timer() - start)

def test(args):
    docs = load_corpus(args.corpus)['docs']
    doc_bow = {}
    for k in docs.keys():
        bows = []
        for idx, count in docs[k].iteritems():
            bows.append((int(idx), count))
        doc_bow[k]= bows
        del docs[k]

    lda = load_model(args.load_model)
    generate_doc_codes(lda, doc_bow, args.output)
    if args.save_topics:
        topics = show_topics(lda)
        write_file(topics, args.save_topics)
        print 'Saved topics file to %s' % args.save_topics

    if args.calc_distinct:
        mean, std = calc_pairwise_cosine(lda)
        print 'Average pairwise angle (pi): %s (%s)' % (mean / math.pi, std / math.pi)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('--corpus', required=True, type=str, help='path to the corpus file')
    parser.add_argument('-nt', '--n_topics', type=int, help='num of topics')
    parser.add_argument('-iter', '--n_iter', type=int, help='num of iterations')
    parser.add_argument('-sm', '--save_model', type=str, default='lda.mod', help='path to the output model')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the trained model')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    parser.add_argument('-st', '--save_topics', type=str, help='path to the output topics file')
    parser.add_argument('-cd', '--calc_distinct', action='store_true', help='calc average pairwise angle')
    args = parser.parse_args()

    if args.train:
        if not args.n_topics:
            raise Exception('n_topics arg needed in training phase')
        train(args)
    else:
        if not args.output:
            raise Exception('output arg needed in test phase')
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
