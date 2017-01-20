'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import numpy as np
from gensim.models import Word2Vec

from ..utils.io_utils import dump_json

def get_doc_codes(model, bow, vocab, size, avg=True):
    vec = np.zeros(size)
    count = 0
    for idx, val in bow.iteritems():
        word = vocab[int(idx)]
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

def doc_word2vec(corpus, vocab, mod_file, output, size=300, avg=True):
    model = load_w2v(mod_file)
    doc_codes = {}
    for key, doc_bow in corpus.iteritems():
        vec = get_doc_codes(model, doc_bow, vocab, size, avg)
        doc_codes[key] = vec.tolist()
    dump_json(doc_codes, output)

    return doc_codes
