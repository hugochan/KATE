'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import re
from random import shuffle
import numpy as np
from collections import Counter, defaultdict

from ..preprocessing.preprocessing import build_vocab, generate_bow, count_words
from ..utils.io_utils import dump_json


class CorpusIterReuters(object):
    def __init__(self, path_list, train_docs, with_docname=False):
        self.path_list = path_list
        self.train_docs = train_docs
        self.with_docname = with_docname

    def __iter__(self):
        shuffle(self.path_list)
        count = 0
        for path in self.path_list:
            with open(path, 'r') as f:
                texts = re.split('\n\s*\n', f.read())[:-1]
                for block in texts:
                    tmp = block.split('\n')
                    did = tmp[0].split(' ')[-1]
                    if not did in self.train_docs:
                        continue
                    words = (' '.join(tmp[2:])).split()
                    count += 1
                    if self.with_docname:
                        yield [words, [did]]
                    else:
                        yield words
        print count

def load_data(path_list, test_split, seed=666):
    '''Loads the Reuters RCV1-v2 newswire dataset.

    @Params
        path_list : a list of file paths
        test_split : fraction of the dataset to be used as test data.
        seed : random seed for sample shuffling.
    '''
    # count the number of times a word appears in a doc
    corpus = {}
    for path in path_list:
        with open(path, 'r') as f:
            texts = re.split('\n\s*\n', f.read())[:-1]
            for block in texts:
                tmp = block.split('\n')
                did = tmp[0].split(' ')[-1]
                count = Counter((' '.join(tmp[2:])).split())
                corpus[did] = dict(count) # doc-word frequency

    corpus = corpus.items()
    np.random.seed(seed)
    np.random.shuffle(corpus)

    n_docs = len(corpus)
    train_data = dict(corpus[:-int(n_docs * test_split)])
    test_data = dict(corpus[-int(n_docs * test_split):])

    return train_data, test_data

def construct_corpus(doc_word_freq, word_freq, training_phase, vocab_dict=None, threshold=5, topn=None):
    if not (training_phase or isinstance(vocab_dict, dict)):
        raise ValueError('vocab_dict must be provided if training_phase is set False')

    if training_phase:
        vocab_dict = build_vocab(word_freq, threshold=threshold, topn=topn)

    docs = generate_bow(doc_word_freq, vocab_dict)
    new_word_freq = dict([(vocab_dict[word], freq) for word, freq in word_freq.iteritems() if word in vocab_dict])

    return docs, vocab_dict, new_word_freq

def construct_train_test_corpus(path_list, test_split, output, threshold=10, topn=20000):
    train_data, test_data = load_data(path_list, test_split)
    train_word_freq = count_words(train_data.values())

    train_docs, vocab_dict, train_word_freq = construct_corpus(train_data, train_word_freq, True, threshold=threshold, topn=topn)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    dump_json(train_corpus, os.path.join(output, 'train.corpus'))
    print 'Generated training corpus'

    test_word_freq = count_words(test_data.values())
    test_docs, _, _ = construct_corpus(test_data, test_word_freq, False, vocab_dict=vocab_dict)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}
    dump_json(test_corpus, os.path.join(output, 'test.corpus'))
    print 'Generated test corpus'

def extract_labels(docs, path, output):
    # it will be fast if docs is a dict instead of a list
    doc_labels = defaultdict(set)
    with open(path, 'r') as f:
        for line in f:
            label, did, _ = line.strip('\n').split()
            if did in docs:
                doc_labels[did].add(label)
    doc_labels = dict([(x, list(y)) for x, y in doc_labels.iteritems()])
    dump_json(doc_labels, output)

    return doc_labels
