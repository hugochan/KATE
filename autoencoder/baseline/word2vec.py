'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import multiprocessing
from gensim.models import word2vec


class Word2Vec(object):
    def __init__(self, dim, min_count=1, sg=1, hs=0, window=5, negative=5, epoches=5):
        super(Word2Vec, self).__init__()
        self.dim = dim
        self.min_count = min_count
        self.sg = sg
        self.hs = hs
        self.window = window
        self.negative = negative
        self.epoches = epoches

    def train(self, corpus):
        self.model = word2vec.Word2Vec(size=self.dim, min_count=self.min_count,\
            window=self.window, workers=multiprocessing.cpu_count(), \
            sg=self.sg, hs=self.hs, negative=self.negative, iter=self.epoches)
        self.model.build_vocab(corpus())
        self.model.train(corpus())

        return self

def save_w2v(model, outfile):
    model.save_word2vec_format(outfile, binary=True)

def load_w2v(mod_file):
    return word2vec.Word2Vec.load_word2vec_format(mod_file, binary=True)
