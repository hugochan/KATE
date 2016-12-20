'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
from os import path
import numpy as np

from autoencoder.core.ae import AutoEncoder, load_model, save_model
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec, vocab_weights
from autoencoder.utils.op_utils import vecnorm, corrupted_matrix
from autoencoder.utils.io_utils import dump_json

def get_topics(ae, vocab, topn=10):
    topics = []
    topic_codes = np.identity(ae.dim)
    dists = ae.decoder.predict(topic_codes)
    dists /= np.sum(dists, axis=1).reshape(ae.dim, 1)
    for idx in range(ae.dim):
        token_idx = np.argsort(dists[idx])[::-1][:topn]
        topic = zip([vocab[x] for x in token_idx], dists[idx][token_idx])
        topics.append(topic)

    return topics

def print_topics(topics):
    for i in range(len(topics)):
        str_topic = ' + '.join(['%s * %s' % (prob, token) for token, prob in topics[i]])
        print 'topic %s:' % i
        print str_topic
        print

def train(args):
    corpus_dir = args.corpus_dir
    n_dim = args.n_dim
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    comp_topk = args.comp_topk
    out_path = args.output

    corpus = load_corpus(corpus_dir)
    vocab, docs, word_freq = corpus['vocab'], corpus['docs'], corpus['word_freq']
    n_vocab = len(vocab)
    n_docs = len(docs)

    X_docs = np.r_[[vecnorm(doc2vec(x, n_vocab), 'logmax1', 0) for x in docs.values()]]
    X_docs_noisy = corrupted_matrix(X_docs, corruption_ratio=.1)

    # Prepare feature_weights for weighted loss
    feature_weights = vecnorm(vocab_weights(vocab, word_freq, max_=100., ratio=.75), 'prob', 0)

    np.random.seed(0)
    train_ratio = .8
    train_idx = np.random.choice(range(n_docs), int(n_docs * train_ratio), replace=False)
    val_idx = list(set(range(n_docs)) - set(train_idx))
    X_train = X_docs[train_idx]
    X_val = X_docs[val_idx]

    # print "Total samples: %s" % n_docs
    # print "Training samples: %s" % X_train.shape[0]
    # print "Validation samples: %s" % X_val.shape[0]

    # X_train_noisy = X_docs_noisy[train_idx]
    # X_val_noisy = X_docs_noisy[val_idx]
    X_train_noisy = X_train
    X_val_noisy = X_val

    ae = AutoEncoder(n_vocab, n_dim, comp_topk=comp_topk, weights_file=args.load_weights, \
            model_save_path=path.join(out_path, 'model_%s.hdf5' % n_dim))
    ae.fit([X_train_noisy, X_train], [X_val_noisy, X_val], nb_epoch=n_epoch, \
            batch_size=batch_size, feature_weights=feature_weights)

    save_model(ae, out_path)
    print 'Saved model.'

def test(args):
    corpus = load_corpus(args.corpus_dir)
    vocab, docs = corpus['vocab'], corpus['docs']
    X_docs = np.r_[[vecnorm(doc2vec(x, len(vocab)), 'logmax1', 0) for x in docs.values()]]
    ae = load_model(args.load_model)

    doc_codes = ae.encoder.predict(X_docs)
    dump_json(dict(zip(docs.keys(), doc_codes.tolist())), path.join(args.output, 'doc_codes.txt'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help='path to the corpus dir')
    parser.add_argument('--train', action='store_true', help='train or test')
    parser.add_argument('-nd', '--n_dim', type=int, default=128, help='num of dimensions')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-o', '--output', type=str, default='./', help='path to the output dir (default current dir)')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the pretrained model')
    parser.add_argument('-lw', '--load_weights', type=str, help='path to the pretrained weights')
    parser.add_argument('-ck', '--comp_topk', type=int, help='saprse topk')
    args = parser.parse_args()

    if args.train:
        train(args)
    else:
        if args.load_model:
            test(args)
        else:
            print 'arg load_model is required when arg train is set False'
            raise

if __name__ == '__main__':
    main()
