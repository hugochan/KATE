'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import

import numpy as np
from utils import *
from autoencoder.varae import VarAutoEncoder

def get_topics(vae, vocab, topn=10):
    topics = []
    topic_codes = np.identity(vae.dim)
    dists = vae.decoder.predict(topic_codes)
    for idx in range(vae.dim):
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
    import sys
    usage = 'python run_ae.py [corpus_path] [n_topics]'
    try:
        corpus_path = sys.argv[1]
        n_topics = int(sys.argv[2])
    except:
        print usage
        sys.exit()

    corpus = load_corpus(corpus_path)

    vocab, docs = corpus['vocab'], corpus['docs']
    n_vocab = len(vocab)
    n_docs = len(docs)

    doc_names = docs.keys()
    X_docs = np.r_[[vecnorm(doc2vec(x, n_vocab), 'logmax1', 0) for x in docs.values()]]

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
    weights = None

    model_save_path = 'mod_files/vae.hdf5'
    vae = VarAutoEncoder(dim=n_topics, nb_epoch=60, batch_size=50, model_save_path=model_save_path)
    try:
        vae.autoencoder = vae.load_mod(sys.argv[3])
        vae.encoder = vae.load_mod(sys.argv[4])
        vae.decoder = vae.load_mod(sys.argv[5])
    except:
        vae.fit([X_train_noisy[:1550], X_train[:1550]], [X_test_noisy[:350], X_test[:350]], feature_weights=feature_weights)


    # print_topics(topics)
    doc_codes = vae.encoder.predict(X_docs[:1950], batch_size=50)
    save_json(dict(zip(doc_names[:1950], doc_codes.tolist())), 'vardoc_codes_logmax1.txt')
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
