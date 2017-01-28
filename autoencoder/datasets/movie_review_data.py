'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import re
from collections import Counter, defaultdict
import numpy as np

from ..preprocessing.preprocessing import build_vocab, generate_bow, tiny_tokenize, init_stopwords
from ..utils.io_utils import dump_json

cached_stop_words = init_stopwords()


class CorpusIterMRD(object):
    def __init__(self, corpus_path, train_docs, stem=True, with_docname=False):
        self.corpus_path = corpus_path
        self.train_docs = train_docs
        self.stem = stem
        self.with_docname = with_docname

    def __iter__(self):
        try:
            with open(self.corpus_path, 'r') as f:
                for line in f:
                    idx, _, subj = line.split('\t')
                    if not idx in self.train_docs:
                        continue
                    words = tiny_tokenize(subj.lower(), stem=self.stem, stop_words=cached_stop_words)
                    if self.with_docname:
                        yield [words, [idx]]
                    else:
                        yield words
        except Exception as e:
            raise e

def load_data(file, test_split, seed=666, stem=False):
    '''Loads the Movie Review Data (https://www.cs.cornell.edu/people/pabo/movie-review-data/).

    @Params
        file : path to the file.
        test_split : fraction of the dataset to be used as test data.
        seed : random seed for sample shuffling.
    '''
    # count the number of times a word appears in a doc
    # cached_stop_words = init_stopwords()

    corpus = {}
    labels = {}
    with open(file, 'r') as f:
        for line in f:
            idx, rating, subj = line.split('\t')
            words = tiny_tokenize(subj.lower(), stem=stem, stop_words=cached_stop_words)
            count = Counter(words)
            corpus[idx] = dict(count) # doc-word frequency
            labels[idx] = float(rating)
    corpus = corpus.items()
    np.random.seed(seed)
    np.random.shuffle(corpus)

    n_docs = len(corpus)
    train_data = dict(corpus[:-int(n_docs * test_split)])
    test_data = dict(corpus[-int(n_docs * test_split):])
    train_labels = dict([(idx, labels[idx]) for idx in train_data.keys()])
    test_labels = dict([(idx, labels[idx]) for idx in test_data.keys()])

    return train_data, train_labels, test_data, test_labels

def count_words(docs):
    # count the number of times a word appears in a corpus
    word_freq = defaultdict(lambda: 0)
    for each in docs:
        for word, val in each.items():
            word_freq[word] += val

    return word_freq

def construct_corpus(doc_word_freq, word_freq, training_phase, vocab_dict=None, threshold=5, topn=None):
    if not (training_phase or isinstance(vocab_dict, dict)):
        raise ValueError('vocab_dict must be provided if training_phase is set False')

    if training_phase:
        vocab_dict = build_vocab(word_freq, threshold=threshold, topn=topn)

    docs = generate_bow(doc_word_freq, vocab_dict)
    new_word_freq = dict([(vocab_dict[word], freq) for word, freq in word_freq.iteritems() if word in vocab_dict])

    return docs, vocab_dict, new_word_freq

def construct_train_test_corpus(file, test_split, output, threshold=10, topn=20000):
    train_data, train_labels, test_data, test_labels = load_data(file, test_split)
    train_word_freq = count_words(train_data.values())

    train_docs, vocab_dict, train_word_freq = construct_corpus(train_data, train_word_freq, True, threshold=threshold, topn=topn)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    dump_json(train_corpus, os.path.join(output, 'train.corpus'))
    print 'Generated training corpus'
    dump_json(train_labels, os.path.join(output, 'train.labels'))
    print 'Generated training labels'

    test_word_freq = count_words(test_data.values())
    test_docs, _, _ = construct_corpus(test_data, test_word_freq, False, vocab_dict=vocab_dict)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}
    dump_json(test_corpus, os.path.join(output, 'test.corpus'))
    print 'Generated test corpus'
    dump_json(test_labels, os.path.join(output, 'test.labels'))
    print 'Generated test labels'
    import pdb;pdb.set_trace()
