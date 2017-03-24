'''
Created on Nov, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse
import math
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

def calc_pairwise_cosine(ae):
    weights = ae.encoder.get_weights()[0]
    weights = unitmatrix(weights, axis=0) # normalize
    n = weights.shape[1]
    score = []
    for i in range(n):
        for j in range(i + 1, n):
            score.append(np.arccos(weights[:, i].dot(weights[:, j])))

    return np.mean(score), np.std(score)

def calc_pairwise_dev(ae):
    # the average squared deviation from 0 (90 degree)
    weights = ae.encoder.get_weights()[0]
    weights = unitmatrix(weights, axis=0) # normalize
    n = weights.shape[1]
    score = 0.
    for i in range(n):
        for j in range(i + 1, n):
            score += (weights[:, i].dot(weights[:, j]))**2

    return np.sqrt(2. * score / n / (n - 1))

def get_similar_words(ae, query_id, vocab, topn=10):
    weights = ae.encoder.get_weights()[0]
    weights = unitmatrix(weights) # normalize
    query = weights[query_id]
    score = query.dot(weights.T)
    vidx = score.argsort()[::-1][:topn]

    return [vocab[idx] for idx in vidx]

def translate_words(ae, query, vocab, revocab, topn=10):
    weights = ae.encoder.get_weights()[0]
    weights = unitmatrix(weights) # normalize
    query_vec = weights[vocab[query[0]]] - weights[vocab[query[1]]] + weights[vocab[query[2]]]
    score = query_vec.dot(weights.T)
    vidx = score.argsort()[::-1][:topn]
    return [revocab[idx] for idx in vidx]

def get_topics(ae, vocab, topn=10):
    topics = []
    weights = ae.encoder.get_weights()[0]
    for idx in range(ae.dim):
        token_idx = np.argsort(weights[:, idx])[::-1][:topn]
        topics.append([vocab[x] for x in token_idx])

    return topics

def get_topics_strength(ae, vocab, topn=10):
    topics = []
    weights = ae.encoder.get_weights()[0]
    for idx in range(ae.dim):
        token_idx = np.argsort(weights[:, idx])[::-1][:topn]
        topics.append([(vocab[x], weights[x, idx]) for x in token_idx])

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
    n_vocab = len(vocab)

    doc_keys = docs.keys()
    X_docs = []
    for k in doc_keys:
        X_docs.append(vecnorm(doc2vec(docs[k], n_vocab), 'logmax1', 0))
        del docs[k]
    X_docs = np.r_[X_docs]

    model = AutoEncoder
    # model = DeepAutoEncoder
    ae = load_model(model, args.load_arch, args.load_weights)

    doc_codes = ae.encoder.predict(X_docs)
    dump_json(dict(zip(doc_keys, doc_codes.tolist())), args.output)
    print 'Saved doc codes file to %s' % args.output

    if args.save_topics:
        topics_strength = get_topics_strength(ae, revdict(vocab), topn=10)
        save_topics_strength(topics_strength, args.save_topics)
        # topics = get_topics(ae, revdict(vocab), topn=10)
        # write_file(topics, args.save_topics)
        print 'Saved topics file to %s' % args.save_topics

    if args.sample_words:
        revocab = revdict(vocab)
        queries = ['weapon', 'christian', 'compani', 'israel', 'law', 'hockey', 'comput', 'space']
        words = []
        for each in queries:
            words.append(get_similar_words(ae, vocab[each], revocab, topn=11))
        write_file(words, args.sample_words)
        print 'Saved sample words file to %s' % args.sample_words
    if args.translate_words:
        revocab = revdict(vocab)
        queries = [['father', 'man', 'woman'], ['mother', 'woman', 'man']]
        for each in queries:
            print each
            print translate_words(ae, each, vocab, revocab, topn=10)
    if args.calc_distinct:
        # mean, std = calc_pairwise_cosine(ae)
        # print 'Average pairwise angle (pi): %s (%s)' % (mean / math.pi, std / math.pi)
        sd = calc_pairwise_dev(ae)
        print 'Average squared deviation from 0 (90 degree): %s' % sd

def save_topics_strength(topics_prob, out_file):
    try:
        with open(out_file, 'w') as datafile:
            for topic in topics_prob:
                datafile.write(' + '.join(["%s * %s" % each for each in topic]) + '\n')
                datafile.write('\n')
    except Exception as e:
        raise e

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='path to the input corpus file')
    parser.add_argument('-o', '--output', type=str, required=True, help='path to the output doc codes file')
    parser.add_argument('-st', '--save_topics', type=str, help='path to the output topics file')
    parser.add_argument('-sw', '--sample_words', type=str, help='path to the output sample words file')
    parser.add_argument('-tw', '--translate_words', action='store_true', help='translate words flag')
    parser.add_argument('-cd', '--calc_distinct', action='store_true', help='calc average pairwise angle')
    parser.add_argument('-la', '--load_arch', type=str, required=True, help='path to the trained arch file')
    parser.add_argument('-lw', '--load_weights', type=str, required=True, help='path to the trained weights file')
    args = parser.parse_args()

    test(args)

if __name__ == '__main__':
    main()
