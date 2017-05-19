'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np

from autoencoder.core.vae import VarAutoEncoder, load_vae_model, save_vae_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec, vocab_weights
from autoencoder.utils.op_utils import vecnorm
from autoencoder.utils.io_utils import dump_json


def train(args):
    corpus = load_corpus(args.input)
    n_vocab, docs = len(corpus['vocab']), corpus['docs']
    corpus.clear() # save memory

    X_docs = []
    for k in docs.keys():
        X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
        del docs[k]

    np.random.seed(0)
    np.random.shuffle(X_docs)
    # X_docs_noisy = corrupted_matrix(np.r_[X_docs], 0.1)

    n_val = args.n_val
    # X_train = np.r_[X_docs[:-n_val]]
    # X_val = np.r_[X_docs[-n_val:]]
    X_train = np.r_[X_docs[:-n_val]]
    del X_docs[:-n_val]
    X_val = np.r_[X_docs]
    del X_docs

    start = timeit.default_timer()

    vae = VarAutoEncoder(n_vocab, args.n_dim, comp_topk=args.comp_topk, ctype=args.ctype, save_model=args.save_model)
    vae.fit([X_train, X_train], [X_val, X_val], nb_epoch=args.n_epoch, batch_size=args.batch_size)

    print 'runtime: %ss' % (timeit.default_timer() - start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-nd', '--n_dim', nargs='*', type=int, help='num of dimensions')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-ck', '--comp_topk', nargs='*', type=int, help='competitive topk')
    parser.add_argument('-ctype', '--ctype', type=str, help='competitive type (kcomp, ksparse, gated_comp)')
    parser.add_argument('-sm', '--save_model', type=str, default='model', help='path to the output model')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
