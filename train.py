'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import timeit
import argparse
from os import path
import numpy as np

from autoencoder.core.ae import AutoEncoder, load_model, save_model
# from autoencoder.core.deepae import DeepAutoEncoder
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec, vocab_weights
from autoencoder.utils.op_utils import vecnorm, add_gaussian_noise, add_masking_noise, add_salt_pepper_noise
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
    if args.noise == 'gs':
        X_docs_noisy = add_gaussian_noise(np.r_[X_docs], 0.1)
    elif args.noise == 'sp':
        X_docs_noisy = add_salt_pepper_noise(np.r_[X_docs], 0.1)
        pass
    elif args.noise == 'mn':
        X_docs_noisy = add_masking_noise(np.r_[X_docs], 0.01)
    else:
        raise 'noise arg should left None or be one of gs, sp or mn'

    n_val = args.n_val
    # X_train = np.r_[X_docs[:-n_val]]
    # X_val = np.r_[X_docs[-n_val:]]
    X_train = np.r_[X_docs[:-n_val]]
    del X_docs[:-n_val]
    X_val = np.r_[X_docs]
    del X_docs

    if args.noise:
        X_train_noisy = X_docs_noisy[:-n_val]
        X_val_noisy = X_docs_noisy[-n_val:]
        print 'added %s noise' % args.noise
    else:
        X_train_noisy = X_train
        X_val_noisy = X_val

    # model = DeepAutoEncoder
    model = AutoEncoder

    start = timeit.default_timer()

    ae = model(n_vocab, args.n_dim, comp_topk=args.comp_topk, weights_file=args.load_weights)
    ae.fit([X_train_noisy, X_train], [X_val_noisy, X_val], nb_epoch=args.n_epoch, \
            batch_size=args.batch_size, feature_weights=None)

    print 'runtime: %ss' % (timeit.default_timer() - start)

    if args.save_model:
        arch_file  = args.save_model + '.arch'
        weights_file  = args.save_model + '.weights'
        save_model(ae, arch_file, weights_file)
    print 'Saved model arch and weights file to %s and %s, respectively.' \
            % (arch_file, weights_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-nd', '--n_dim', type=int, default=128, help='num of dimensions (default 128)')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-nv', '--n_val', type=int, default=1000, help='size of validation set (default 1000)')
    parser.add_argument('-ck', '--comp_topk', type=int, help='competitive topk')
    parser.add_argument('-lw', '--load_weights', type=str, help='path to the pretrained weights file')
    parser.add_argument('-sm', '--save_model', type=str, default='model', help='path to the output model')
    parser.add_argument('--noise', type=str, help='noise type: gs for Gaussian noise, sp for salt-and-pepper or mn for masking noise')
    args = parser.parse_args()

    if args.noise and not args.noise in ['gs', 'sp', 'mn']:
        raise 'noise arg should left None or be one of gs, sp or mn'
    train(args)

if __name__ == '__main__':
    main()
