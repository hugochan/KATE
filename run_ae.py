'''
Created on Nov, 2016

@author: hugo

'''

import os
import numpy as np
from utils import *
from ae import AutoEncoder

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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus_dir', type=str, help='path to the corpus dir')
    parser.add_argument('n_dim', type=int, help='num of dimensions')
    parser.add_argument('-ne', '--n_epoch', type=int, default=100, help='num of epoches (default 100)')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('-o', '--output', type=str, default='./', help='path to the output dir (default current dir)')
    parser.add_argument('-lm', '--load_model', type=str, help='path to the pretrained model')
    parser.add_argument('-lw', '--load_weights', type=str, help='path to the pretrained weights')
    parser.add_argument('-sw', '--save_weights', action='store_true', help='save weights flag')
    parser.add_argument('-sk', '--sparse_topk', type=int, default=None, help='saprse topk')
    parser.add_argument('-sa', '--sparse_alpha', type=float, default=None, help='saprse alpha')
    args = parser.parse_args()

    corpus_dir = args.corpus_dir
    n_dim = args.n_dim
    n_epoch = args.n_epoch
    batch_size = args.batch_size
    out_path = args.output
    model_path = args.load_model
    sparse_topk = args.sparse_topk
    sparse_alpha = args.sparse_alpha

    corpus = load_corpus(corpus_dir)
    vocab, docs = corpus['vocab'], corpus['docs']
    # docs = dict(docs.items()[:5000])
    n_vocab = len(vocab)
    n_docs = len(docs)

    doc_names = docs.keys()
    X_docs = np.r_[[vecnorm(doc2vec(x, n_vocab), 'logmax1', 0) for x in docs.values()]]


    # feature_weights = vocab_weights_tfidf(vocab, corpus['word_freq'], docs.values(), max_=100., ratio=.6)
    feature_weights = vocab_weights(vocab, corpus['word_freq'], max_=100., ratio=.75)
    feature_weights = vecnorm(feature_weights, 'prob', 0)

    X_docs_noisy = corrupted_matrix(X_docs, corruption_ratio=.1)


    np.random.seed(0)
    train_ratio = .8
    train_idx = np.random.choice(range(n_docs), int(n_docs * train_ratio), replace=False)
    test_idx = list(set(range(n_docs)) - set(train_idx))
    X_train = X_docs[train_idx]
    X_test = X_docs[test_idx]

    print "total samples: %s" % n_docs
    print "training samples: %s" % X_train.shape[0]
    print "test samples: %s" % X_test.shape[0]

    # X_train_noisy = X_docs_noisy[train_idx]
    # X_test_noisy = X_docs_noisy[test_idx]
    X_train_noisy = X_train
    X_test_noisy = X_test


    ae = AutoEncoder(dim=n_dim, nb_epoch=n_epoch, batch_size=batch_size, model_save_path=os.path.join(out_path, 'model.hdf5'))
    ae.fit([X_train_noisy, X_train], [X_test_noisy, X_test], sparse_topk=sparse_topk, sparse_alpha=sparse_alpha,\
            feature_weights=feature_weights, init_weights=None, weights_file=args.load_weights)

    if args.save_weights:
        ae.autoencoder.save_weights(os.path.join(out_path, 'weights_%s.h5' % n_dim))

    doc_codes = ae.encoder.predict(X_docs)
    save_json(dict(zip(doc_names, doc_codes.tolist())), os.path.join(out_path, 'doc_codes.txt'))
    import pdb;pdb.set_trace()
    # topics = get_topics(ae, revdict(vocab), topn=10)

if __name__ == '__main__':
    main()
