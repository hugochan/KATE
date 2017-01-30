'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import multiprocessing
from gensim.models import Doc2Vec


class MyDoc2Vec(object):
    def __init__(self, dim, hs=0, window=5, negative=5, epoches=5, dm=1, dm_concat=1):
        super(MyDoc2Vec, self).__init__()
        self.dim = dim
        self.hs = hs
        self.window = window
        self.negative = negative
        self.epoches = epoches
        self.dm = dm
        self.dm_concat = dm_concat

    def train(self, corpus):
        self.model = Doc2Vec(min_count=1, window=self.window, size=self.dim, \
            workers=multiprocessing.cpu_count(), hs=self.hs,\
            negative=self.negative, iter=1, dm=self.dm, dm_concat=self.dm_concat)
        self.model.build_vocab(corpus())
        for each in range(self.epoches):
            self.model.train(corpus())

        return self

def predict(model, corpus):
    doc_codes = {}
    for doc_words, doc_name in corpus():
        doc_codes[doc_name[0]] = model.infer_vector(doc_words).tolist()

    return doc_codes

def save_doc2vec(model, outfile):
    model.save(outfile)

def load_doc2vec(mod_file):
    return Doc2Vec.load(mod_file)
