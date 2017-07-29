'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel

from ..utils.io_utils import dump_json
from ..utils.op_utils import unitmatrix


def train_lda(corpus, vocab_dict, n_topics, n_iter, save_model):
    lda = LdaModel(corpus, num_topics=n_topics, id2word=vocab_dict, \
        passes=n_iter, minimum_probability=1e-3)
    lda.save(save_model)

    return lda

def generate_doc_codes(model, corpus, output):
    model.minimum_probability = 1e-3
    n_topics = model.num_topics
    doc_codes = {}
    for key, doc_bow in corpus.iteritems():
        code = np.zeros(n_topics)
        for idx, val in model[doc_bow]:
            code[idx] = val
        doc_codes[key] = code.tolist()
    dump_json(doc_codes, output)

    return doc_codes

def show_topics(model, n_words_per_topic=10):
    n_topics = model.num_topics
    topics = [zip(*model.show_topic(i, n_words_per_topic))[0] for i in range(n_topics)]

    return topics

def show_topics_prob(model, n_words_per_topic=10):
    n_topics = model.num_topics
    topics = [model.show_topic(i, n_words_per_topic) for i in range(n_topics)]

    return topics

def calc_pairwise_cosine(model):
    n = model.num_topics
    weights = model.state.get_lambda()
    weights = np.apply_along_axis(lambda x: x / x.sum(), 1, weights) # get dist.
    weights = unitmatrix(weights) # normalize
    score = []
    for i in range(n):
        for j in range(i + 1, n):
            score.append(np.arccos(weights[i].dot(weights[j])))

    return np.mean(score), np.std(score)

def calc_pairwise_dev(model):
    # the average squared deviation from 0 (90 degree)
    n = model.num_topics
    weights = model.state.get_lambda()
    weights = np.apply_along_axis(lambda x: x / x.sum(), 1, weights) # get dist.
    weights = unitmatrix(weights) # normalize
    score = 0.
    for i in range(n):
        for j in range(i + 1, n):
            score += (weights[i].dot(weights[j]))**2

    return np.sqrt(2. * score / n / (n - 1))

def load_model(model_file):
    return LdaModel.load(model_file)
