'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np

from autoencoder.core.vae import VarAutoEncoder, load_vae_model
from autoencoder.core.deepae import DeepAutoEncoder
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, revdict
from autoencoder.utils.io_utils import dump_json, write_file

# def get_topics(vae, vocab, topn=10):
#     topics = []
#     weights = vae.encoder.get_weights()[0]
#     for idx in range(ae.dim):
#         token_idx = np.argsort(weights[:, idx])[::-1][:topn]
#         topics.append([vocab[x] for x in token_idx])

#     return topics

# def print_topics(topics):
#     for i in range(len(topics)):
#         str_topic = ' + '.join(['%s * %s' % (prob, token) for token, prob in topics[i]])
#         print 'topic %s:' % i
#         print str_topic
#         print

def test(args):
    corpus = load_corpus(args.input)
    vocab, docs = corpus['vocab'], corpus['docs']
    n_vocab = len(vocab)

    doc_keys = docs.keys()
    X_docs = []
    for k in doc_keys:
        X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
        del docs[k]
    X_docs = np.r_[X_docs]

    vae = load_vae_model(args.load_model)

    doc_codes = vae.predict(X_docs)
    dump_json(dict(zip(doc_keys, doc_codes.tolist())), args.output)
    print 'Saved doc codes file to %s' % args.output

    # if args.save_topics:
    #     topics = get_topics(vae, revdict(vocab), topn=10)
    #     write_file(topics, args.save_topics)
    #     print 'Saved topics file to %s' % args.save_topics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to the output doc codes file')
    parser.add_argument('-lm', '--load_model', type=str, required=True, help='path to the trained model file')
    # parser.add_argument('-st', '--save_topics', type=str, help='path to the output topics file')
    args = parser.parse_args()

    test(args)

if __name__ == '__main__':
    main()
