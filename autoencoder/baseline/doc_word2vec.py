'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import numpy as np
from gensim.models import Word2Vec

from ..utils.io_utils import dump_json

def get_doc_codes(model, bow, vocab, avg=True):
    vec = np.zeros(model.vector_size)
    count = 0
    for idx in bow:
        word = vocab[int(idx)]
        val = bow[idx]
        if word in model:
            vec += model[word] * val
            count += val
        elif word.title() in model:
            vec += model[word.title()] * val
            count += val
        elif word.upper() in model:
            vec += model[word.upper()] * val
            count += val

    return vec / count if avg else vec

def load_w2v(file):
    model = Word2Vec.load_word2vec_format(file, binary=True)
    return model

def doc_word2vec(model, corpus, vocab, output, avg=True):
    doc_codes = {}
    for key, bow in corpus.iteritems():
        vec = get_doc_codes(model, bow, vocab, avg)
        doc_codes[key] = vec.tolist()
    dump_json(doc_codes, output)

    return doc_codes

def get_similar_words(model, query, topn=10):
    return zip(*model.most_similar(query, topn=topn))[0]
