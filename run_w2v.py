'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import timeit
import numpy as np

from autoencoder.baseline.word2vec import Word2Vec, save_w2v, load_w2v
from autoencoder.baseline.doc_word2vec import doc_word2vec
from autoencoder.utils.io_utils import load_json, dump_json, write_file
from autoencoder.preprocessing.preprocessing import load_corpus
# from autoencoder.datasets.reuters import CorpusIterReuters
from autoencoder.datasets.the20news import CorpusIter20News
# from autoencoder.datasets.movie_review_data import CorpusIterMRD
# from autoencoder.datasets.wiki10plus import CorpusIterWiki10plus


def train(args):
    vocab = load_json(args.vocab)
    # import pdb;pdb.set_trace()
    # load corpus
    corpus = CorpusIter20News(args.corpus[0], recursive=True, stem=True, with_docname=False)
    # corpus = CorpusIterMRD(args.corpus[0], load_json(args.docnames), stem=True, with_docname=False)
    # corpus = CorpusIterWiki10plus(args.corpus[0], load_json(args.docnames), stem=True, with_docname=False)
    # corpus = CorpusIterReuters(args.corpus, load_json(args.docnames), with_docname=False)
    # print len([1 for x in corpus])
    corpus_iter = lambda: ([word for word in sentence if word in vocab] for sentence in corpus)
    w2v = Word2Vec(args.n_dim, window=args.window_size, \
        negative=args.negative, epoches=args.n_epoch)

    start = timeit.default_timer()
    w2v.train(corpus_iter)
    print 'runtime: %ss' % (timeit.default_timer() - start)

    save_w2v(w2v.model, args.save_model)
    import pdb;pdb.set_trace()

def test(args):
    corpus = load_corpus(args.corpus[0])
    docs, vocab_dict = corpus['docs'], corpus['vocab']
    doc_codes = doc_word2vec(docs, revdict(vocab_dict), args.load_model, args.output, avg=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('--corpus', nargs='*', required=True, type=str, help='path to the corpus dir (in training phase) or file (in test phase)')
    parser.add_argument('-doc', '--docnames', type=str, help='path to the docnames file (in training phase)')
    parser.add_argument('--vocab', required=True, type=str, help='path to the vocab file')
    parser.add_argument('-ne', '--n_epoch', required=True, type=int, help='num of epoches')
    parser.add_argument('-nd', '--n_dim', type=int, help='num of dimensions')
    parser.add_argument('-ws', '--window_size', required=True, type=int, help='window size')
    parser.add_argument('-neg', '--negative', required=True, type=int, help='num of negative samples')
    parser.add_argument('-sm', '--save_model', type=str, default='w2v.mod', help='path to the output model')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the trained model')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    if args.train:
        if not args.n_dim:
            raise Exception('n_dim arg needed in training phase')
        train(args)
    else:
        if not args.output:
            raise Exception('output arg needed in test phase')
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
