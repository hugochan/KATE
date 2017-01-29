'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
from random import shuffle
from collections import defaultdict


from ..preprocessing.preprocessing import get_all_files, init_stopwords, tiny_tokenize

cached_stop_words = init_stopwords()


class CorpusIter20News(object):
    def __init__(self, corpus_path, recursive=False, stem=True, with_docname=False):
        self.stem = stem
        self.with_docname = with_docname
        self.files = get_all_files(corpus_path, recursive)

    def __iter__(self):
        shuffle(self.files)
        count = 0
        for filename in self.files:
            try:
                with open(filename, 'r') as fp:
                    text = fp.read().lower()
                    # remove punctuations, stopwords and *unnecessary digits*, stemming
                    words = tiny_tokenize(text, self.stem, cached_stop_words)
                    count += 1
                    if self.with_docname:
                        parent_name, child_name = os.path.split(filename)
                        doc_name = os.path.split(parent_name)[-1] + '_' + child_name
                        yield [words, [doc_name]]
                    else:
                        yield words
            except Exception as e:
                raise e
        print count
