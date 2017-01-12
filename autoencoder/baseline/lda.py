'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import numpy as np
from gensim import corpora
from gensim.models.ldamodel import LdaModel

from ..utils.io_utils import dump_json

def train_lda(corpus, vocab_dict, n_topics, save_model):
    lda = LdaModel(corpus, num_topics=n_topics, id2word=vocab_dict)
    lda.save(save_model)

def generate_doc_codes(model, corpus, n_topics, output):
    doc_codes = {}
    for key, doc_bow in corpus.iteritems():
        code = np.zeros(n_topics)
        for idx, val in model[doc_bow]:
            code[idx] = val
        doc_codes[key] = code.tolist()
    dump_json(doc_codes, output)
    return doc_codes

def show_topics(model, n_topics, n_words_per_topic=10):
    topics = [zip(*model.show_topic(i, n_words_per_topic))[0] for i in range(n_topics)]
    return topics

def load_model(model_file):
    return LdaModel.load(model_file)
