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

    size = X_train.shape[0] - X_train.shape[0] % args.batch_size
    vae = VarAutoEncoder(n_vocab, args.n_intermediate_dim, args.n_dim, args.batch_size, weights_file=args.load_weights)
    vae.fit([X_train[:size], X_train[:size]], [X_val, X_val], nb_epoch=args.n_epoch)

    print 'runtime: %ss' % (timeit.default_timer() - start)

    if args.save_model:
        arch_file  = args.save_model + '.arch'
        weights_file  = args.save_model + '.weights'
        save_vae_model(vae, arch_file, weights_file)
    print 'Saved model arch and weights file to %s and %s, respectively.' \
            % (arch_file, weights_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-nid', '--n_intermediate_dim', type=int, default=512, help='num of intermediate dimensions (default 512)')
    parser.add_argument('-nd', '--n_dim', type=int, default=128, help='num of dimensions (default 128)')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-lw', '--load_weights', type=str, help='path to the pretrained weights file')
    parser.add_argument('-sm', '--save_model', type=str, default='model', help='path to the output model')
    args = parser.parse_args()

    train(args)

if __name__ == '__main__':
    main()
