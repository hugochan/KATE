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
from autoencoder.baseline.lda import train_lda, generate_doc_codes, load_model, show_topics, show_topics_prob, calc_pairwise_cosine, calc_pairwise_dev
from autoencoder.testing.visualize import word_cloud
from autoencoder.utils.op_utils import unitmatrix


def train(args):
    corpus = load_corpus(args.corpus)
    docs, vocab_dict = corpus['docs'], corpus['vocab']
    doc_bow = []
    doc_keys = docs.keys()
    for k in doc_keys:
        bows = []
        for idx, count in docs[k].iteritems():
            bows.append((int(idx), count))
        doc_bow.append(bows)
        del docs[k]
    vocab_dict = dict([(int(y), x) for x, y in vocab_dict.iteritems()])

    n_samples = len(doc_bow)
    doc_bow = np.array(doc_bow)
    np.random.seed(0)
    val_idx = np.random.choice(range(n_samples), args.n_val, replace=False)
    train_idx = list(set(range(n_samples)) - set(val_idx))
    dbow_train = doc_bow[train_idx].tolist()
    dbow_val = doc_bow[val_idx].tolist()
    del doc_bow

    start = timeit.default_timer()
    lda = train_lda(dbow_train, vocab_dict, args.n_topics, args.n_iter, args.save_model)
    print 'runtime: %ss' % (timeit.default_timer() - start)

    if args.output:
        doc_keys = np.array(doc_keys)
        generate_doc_codes(lda, dict(zip(doc_keys[train_idx].tolist(), dbow_train)), args.output + '.train')
        generate_doc_codes(lda, dict(zip(doc_keys[val_idx].tolist(), dbow_val)), args.output + '.val')
        print 'Saved doc codes file to %s and %s' % (args.output + '.train', args.output + '.val')

def test(args):
    corpus = load_corpus(args.corpus)
    vocab, docs = corpus['vocab'], corpus['docs']
    doc_bow = {}
    for k in docs.keys():
        bows = []
        for idx, count in docs[k].iteritems():
            bows.append((int(idx), count))
        doc_bow[k]= bows
        del docs[k]

    lda = load_model(args.load_model)
    generate_doc_codes(lda, doc_bow, args.output)
    print 'Saved doc codes file to %s' % args.output


    if args.word_clouds:
        queries = ['interest', 'trust', 'cash', 'payment', 'rate', 'price', 'stock', 'share', 'award', 'risk', 'security', 'bank', 'company',\
             'service', 'grant', 'agreement', 'proxy', 'loan', 'capital', 'asset', 'bonus', 'shareholder', 'income', 'financial', 'net', 'purchase',\
             'position', 'management', 'loss', 'salary', 'stockholder', 'due', 'business', 'transaction', 'govern', 'trading',\
             'tax', 'march', 'june']
        # queries = ['interest', 'trust', 'cash', 'payment', 'rate', 'price', 'stock', 'share', \
        #      'award', 'risk', 'security', 'bank', 'company', 'service', 'grant', 'agreement', \
        #      'proxy', 'loan', 'capital', 'asset', 'bonus', 'shareholder', 'income', 'financial', \
        #      'net', 'purchase', 'position', 'management', 'loss', 'salary', 'stockholder', 'due', \
        #      'business', 'transaction', 'govern', 'trading', 'tax', 'three', 'four', 'five', \
        #      'eleven', 'thirteen', 'fifteen', 'eighteen', 'twenty']
        weights = lda.state.get_lambda()
        weights = np.apply_along_axis(lambda x: x / x.sum(), 1, weights) # get dist.
        # weights = unitmatrix(weights, axis=1) # normalize
        word_cloud(weights.T, vocab, queries, save_file=args.word_clouds)

        print 'Saved word clouds file to %s' % args.word_clouds

    if args.save_topics:
        topics_prob = show_topics_prob(lda)
        save_topics_prob(topics_prob, args.save_topics)
        # topics = show_topics(lda)
        # write_file(topics, args.save_topics)
        print 'Saved topics file to %s' % args.save_topics

    if args.calc_distinct:
        # mean, std = calc_pairwise_cosine(lda)
        # print 'Average pairwise angle (pi): %s (%s)' % (mean / math.pi, std / math.pi)
        sd = calc_pairwise_dev(lda)
        print 'Average squared deviation from 0 (90 degree): %s' % sd

def save_topics_prob(topics_prob, out_file):
    try:
        with open(out_file, 'w') as datafile:
            for topic in topics_prob:
                datafile.write(' + '.join(["%s * %s" % each for each in topic]) + '\n')
                datafile.write('\n')
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('--corpus', required=True, type=str, help='path to the corpus file')
    parser.add_argument('-nt', '--n_topics', type=int, help='num of topics')
    parser.add_argument('-iter', '--n_iter', type=int, help='num of iterations')
    parser.add_argument('-sm', '--save_model', type=str, default='lda.mod', help='path to the output model')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the trained model')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    parser.add_argument('-st', '--save_topics', type=str, help='path to the output topics file')
    parser.add_argument('-wc', '--word_clouds', type=str, help='path to the output word clouds file')
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
