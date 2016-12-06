'''
Created on Nov, 2016

@author: hugo

'''

import numpy as np
from utils import *
from varae import VarAutoEncoder

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

    X_docs = np.r_[[vecnorm(doc2vec(x, n_vocab), 'prob', 1e-5) for x in docs]]
    # X_docs = np.r_[[doc2vec(x, n_vocab) for x in docs]]

    train_ratio = .8
    train_idx = np.random.choice(range(n_docs), int(n_docs * train_ratio), replace=False)[:1500]
    test_idx = list(set(range(n_docs)) - set(train_idx))[:450]
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

    vae = VarAutoEncoder(dim=n_topics, nb_epoch=500, batch_size=150)
    try:
        vae.autoencoder = vae.load_mod(sys.argv[3])
        vae.encoder = vae.load_mod(sys.argv[4])
        vae.decoder = vae.load_mod(sys.argv[5])
    except:
        vae.fit([X_train_noisy, X_train], [X_test_noisy, X_test])
        # vae.save_all([[ae.autoencoder, 'autoenoder_s.mod'], [ae.encoder, 'encoder_s.mod'], [ae.decoder, 'decoder_s.mod']])

    # note that we take them from the *test* set
    # encoded_X = vae.encoder.predict(X_test_noisy)
    # decoded_X = vae.decoder.predict(encoded_X)

    import pdb;pdb.set_trace()
    topics = get_topics(vae, revdict(vocab), topn=10)
    print_topics(topics)

if __name__ == '__main__':
    main()
