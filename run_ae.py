'''
Created on Nov, 2016

@author: hugo

'''

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
    # X_docs = np.r_[[doc2vec(x, n_vocab) for x in docs]]


    feature_weights = vocab_weights(vocab, corpus['word_freq'], max_=100., ratio=.75)
    feature_weights = vecnorm(feature_weights, 'prob', 0)

    X_docs_noisy = corrupted_matrix(X_docs, corruption_ratio=.25)


    np.random.seed(0)
    train_ratio = .8
    train_idx = np.random.choice(range(n_docs), int(n_docs * train_ratio), replace=False)
    test_idx = list(set(range(n_docs)) - set(train_idx))
    X_train = X_docs[train_idx]
    X_test = X_docs[test_idx]

    # X_train = np.random.randn(n_docs, n_vocab)
    # X_train[X_train < 0] = 0
    # X_train /= np.sum(X_train, axis=1).reshape(n_docs, 1)
    # X_test = np.random.randn(n_docs, n_vocab)
    # X_test[X_test < 0] = 0
    # X_test /= np.sum(X_test, axis=1).reshape(n_docs, 1)

    print "total samples: %s" % n_docs
    print "training samples: %s" % X_train.shape[0]
    print "test samples: %s" % X_test.shape[0]

    X_train_noisy = X_docs_noisy[train_idx]
    X_test_noisy = X_docs_noisy[test_idx]
    # X_train_noisy = X_train
    # X_test_noisy = X_test
    weights = None
    # try:
    #     import json
    #     topic_vocab_dist = json.load(open(sys.argv[3], 'r'))
    #     weights = init_weights(topic_vocab_dist, vocab)
    # except Exception as e:
    #     print e
    #     weights = np.random.randn(n_vocab, n_topics) / np.sqrt(n_vocab)

    ae = AutoEncoder(dim=n_topics, nb_epoch=100, batch_size=100)
    try:
        ae.autoencoder = ae.load_mod(sys.argv[3])
        ae.encoder = ae.load_mod(sys.argv[4])
        ae.decoder = ae.load_mod(sys.argv[5])
    except:
        ae.fit([X_train_noisy, X_train], [X_test_noisy, X_test], feature_weights=feature_weights, init_weights=weights)
        # ae.save_all([[ae.autoencoder, 'autoenoder_nbias.mod'], [ae.encoder, 'encoder_nbias.mod'], [ae.decoder, 'decoder_nbias.mod']])

    # note that we take them from the *test* set
    # encoded_X = ae.encoder.predict(X_test_noisy)
    # decoded_X = ae.decoder.predict(encoded_X)

    # print_topics(topics)
    doc_codes = ae.encoder.predict(X_docs)
    save_json(dict(zip(doc_names, doc_codes.tolist())), 'doc_codes_logmax1.txt')
    import pdb;pdb.set_trace()
    # topics = get_topics(ae, revdict(vocab), topn=10)

if __name__ == '__main__':
    main()
