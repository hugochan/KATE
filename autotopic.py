'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np
from utils import *
from autoencoder import AutoEncoder

def get_topics(topic_token_dist, vocab, topn=10):
    topics = []
    for idx in range(topic_token_dist.shape[0]):
        dist = vecnorm(topic_token_dist[idx], 'prob', 0)
        token_idx = np.argsort(dist)[::-1][:topn]
        topic = zip([vocab[x] for x in token_idx], dist[token_idx])
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
    usage = 'python autotopic.py [corpus_path] [n_topics]'
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

    X_docs = np.r_[[vecnorm(doc2vec(x, n_vocab), 'prob', 0) for x in docs]]
    # X_docs = np.r_[[doc2vec(x, n_vocab) for x in docs]]

    train_ratio = .8
    train_idx = np.random.choice(range(n_docs), int(n_docs * train_ratio), replace=False)
    test_idx = list(set(range(n_docs)) - set(train_idx))
    X_train = X_docs[train_idx]
    X_test = X_docs[test_idx]
    print "total samples: %s" % n_docs
    print "training samples: %s" % X_train.shape[0]
    print "test samples: %s" % X_test.shape[0]
    # import pdb;pdb.set_trace()

    # X_train_noisy = corrupted_matrix(X_train, corruption_ratio=.1)
    # X_test_noisy = corrupted_matrix(X_test, corruption_ratio=.1)
    X_train_noisy = X_train
    X_test_noisy = X_test

    ae = AutoEncoder(dim=n_topics, nb_epoch=100, batch_size=100).fit([X_train_noisy, X_train], [X_test_noisy, X_test])
    # encoded_X = ae.encoder.predict(X_test_noisy)
    # decoded_X = ae.decoder.predict(encoded_X)

    topics = get_topics(ae.decoder.get_weights()[0], revdict(vocab), topn=10)
    print_topics(topics)
    # ae.save_mod('autotopic.mod')
    import pdb;pdb.set_trace()

if __name__ == '__main__':
    main()
