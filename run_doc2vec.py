'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import timeit
import numpy as np
from gensim.models.doc2vec import TaggedDocument

from autoencoder.baseline.doc2vec import MyDoc2Vec, save_doc2vec, load_doc2vec, predict
from autoencoder.utils.io_utils import load_json, dump_json, write_file
from autoencoder.preprocessing.preprocessing import load_corpus
from autoencoder.datasets.the20news import CorpusIter20News
# from autoencoder.datasets.movie_review_data import CorpusIterMRD
# from autoencoder.datasets.wiki10plus import CorpusIterWiki10plus
# from autoencoder.datasets.reuters import CorpusIterReuters


def train(args):
    vocab = load_json(args.vocab)
    # import pdb;pdb.set_trace()
    # load corpus
    corpus = CorpusIter20News(args.corpus[0], recursive=True, stem=True, with_docname=True)
    # corpus = CorpusIterMRD(args.corpus[0], load_json(args.docnames), stem=True, with_docname=True)
    # corpus = CorpusIterWiki10plus(args.corpus[0], load_json(args.docnames), stem=True, with_docname=True)
    # corpus = CorpusIterReuters(args.corpus, load_json(args.docnames), with_docname=True)
    corpus_iter = lambda: (TaggedDocument([word for word in sentence if word in vocab], tag) for sentence, tag in corpus)

    d2v = MyDoc2Vec(args.n_dim, window=args.window_size, \
        negative=args.negative, epoches=args.n_epoch, dm_concat=1)

    start = timeit.default_timer()
    d2v.train(corpus_iter)
    print 'runtime: %ss' % (timeit.default_timer() - start)

    save_doc2vec(d2v.model, args.save_model)
    import pdb;pdb.set_trace()


def test(args):
    vocab = load_json(args.vocab)
    # load corpus
    corpus = CorpusIter20News(args.corpus[0], recursive=True, stem=True, with_docname=True)
    # corpus = CorpusIterMRD(args.corpus[0], load_json(args.docnames), stem=True, with_docname=True)
    # corpus = CorpusIterWiki10plus(args.corpus[0], load_json(args.docnames), stem=True, with_docname=True)
    # corpus = CorpusIterReuters(args.corpus, load_json(args.docnames), with_docname=True)
    corpus_iter = lambda: (TaggedDocument([word for word in sentence if word in vocab], tag) for sentence, tag in corpus)

    d2v = load_doc2vec(args.load_model)
    doc_codes = predict(d2v, corpus_iter)
    dump_json(doc_codes, args.output)
    import pdb;pdb.set_trace()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train flag')
    parser.add_argument('--corpus', nargs='*', required=True, type=str, help='path to the corpus dir')
    parser.add_argument('-doc', '--docnames', type=str, help='path to the docnames file (in training phase)')
    parser.add_argument('--vocab', required=True, type=str, help='path to the vocab file')
    parser.add_argument('-ne', '--n_epoch', type=int, help='num of epoches')
    parser.add_argument('-nd', '--n_dim', type=int, help='num of dimensions')
    parser.add_argument('-ws', '--window_size', type=int, help='window size')
    parser.add_argument('-neg', '--negative', type=int, help='num of negative samples')
    parser.add_argument('-sm', '--save_model', type=str, default='w2v.mod', help='path to the output model')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the trained model')
    parser.add_argument('-o', '--output', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    if args.train:
        if not args.n_epoch or not args.n_dim or \
            not args.window_size or not args.negative:
            raise Exception('n_epoch, n_dim, window_size and negative args needed in test phase')
        train(args)
    else:
        if not args.output:
            raise Exception('output arg needed in test phase')
        if not args.load_model:
            raise Exception('load_model arg needed in test phase')
        test(args)

if __name__ == '__main__':
    main()
