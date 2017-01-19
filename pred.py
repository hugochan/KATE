'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import numpy as np

from autoencoder.core.ae import AutoEncoder, load_model
# from autoencoder.core.deepae import DeepAutoEncoder
from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.op_utils import vecnorm, revdict, unitmatrix #, corrupted_matrix
from autoencoder.utils.io_utils import dump_json, write_file

# def get_topics(ae, vocab, topn=10):
#     topics = []
#     topic_codes = np.identity(ae.dim)
#     dists = ae.decoder.predict(topic_codes)
#     dists /= np.sum(dists, axis=1).reshape(ae.dim, 1)
#     for idx in range(ae.dim):
#         token_idx = np.argsort(dists[idx])[::-1][:topn]
#         topic = zip([vocab[x] for x in token_idx], dists[idx][token_idx])
#         topics.append(topic)

#     return topics

def get_similar_words(ae, query_id, vocab, topn=10):
    weights = ae.encoder.get_weights()[0]
    weights = unitmatrix(weights) # normalize
    query = weights[query_id]
    score = query.dot(weights.T)
    vidx = score.argsort()[::-1][:topn]
    return [vocab[idx] for idx in vidx]

def get_topics(ae, vocab, topn=10):
    topics = []
    weights = ae.encoder.get_weights()[0]
    for idx in range(ae.dim):
        token_idx = np.argsort(weights[:, idx])[::-1][:topn]
        topics.append([vocab[x] for x in token_idx])

    return topics

def print_topics(topics):
    for i in range(len(topics)):
        str_topic = ' + '.join(['%s * %s' % (prob, token) for token, prob in topics[i]])
        print 'topic %s:' % i
        print str_topic
        print

def test(args):
    corpus = load_corpus(args.input)
    vocab, docs = corpus['vocab'], corpus['docs']

    X_docs = []
    for k in docs.keys():
        X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
        del docs[k]
    X_docs = np.r_[X_docs]

    model = AutoEncoder
    # model = DeepAutoEncoder
    ae = load_model(model, args.load_arch, args.load_weights)

    doc_codes = ae.encoder.predict(X_docs)
    dump_json(dict(zip(docs.keys(), doc_codes.tolist())), args.output)
    print 'Saved doc codes file to %s' % args.output

    if args.save_topics:
        topics = get_topics(ae, revdict(vocab), topn=10)
        write_file(topics, args.save_topics)
        print 'Saved topics file to %s' % args.save_topics

    if args.sample_words:
        queries = ['weapon', 'christian', 'compani', 'israel', 'law', 'hockey', 'comput', 'space']
        words = []
        for each in queries:
            words.append(get_similar_words(ae, vocab[each], revdict(vocab), topn=11))
        write_file(words, args.sample_words)
        print 'Saved sample words file to %s' % args.sample_words


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to the output doc codes file')
    parser.add_argument('-st', '--save_topics', type=str, help='path to the output topics file')
    parser.add_argument('-sw', '--sample_words', type=str, help='path to the output sample words file')
    parser.add_argument('-la', '--load_arch', type=str, required=True, help='path to the trained arch file')
    parser.add_argument('-lw', '--load_weights', type=str, required=True, help='path to the trained weights file')
    args = parser.parse_args()

    test(args)

if __name__ == '__main__':
    main()
