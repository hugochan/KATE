'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import re
import numpy as np
from random import shuffle
from collections import Counter

from ..preprocessing.preprocessing import init_stopwords, tiny_tokenize_xml, tiny_tokenize, get_all_files, count_words
from ..datasets.reuters import construct_corpus
from ..utils.io_utils import dump_json

pattern = r'>([^<>]+)<'
prog = re.compile(pattern)
cached_stop_words = init_stopwords()


class CorpusIterWiki10plus(object):
    def __init__(self, corpus_dir, train_docs, stem=True, with_docname=False):
        self.stem = stem
        self.train_docs = train_docs
        self.with_docname = with_docname
        self.files = get_all_files(corpus_dir, False)

    def __iter__(self):
        shuffle(self.files)
        count = 0
        for filename in self.files:
            doc_name = os.path.basename(filename)
            if not doc_name in self.train_docs:
                continue
            try:
                with open(filename, 'r') as fp:
                    count += 1
                    text = fp.read().lower()
                    # remove punctuations, stopwords and *unnecessary digits*, stemming
                    words = tiny_tokenize(text, self.stem, cached_stop_words)
                    if self.with_docname:
                        yield [words, [doc_name]]
                    else:
                        yield words
            except Exception as e:
                raise e
        print count

def extract_contents(text, out_file):
    if not isinstance(text, unicode):
        text = text.decode('utf-8')
    contents = ' '.join(prog.findall(text))
    contents = tiny_tokenize_xml(contents, False, cached_stop_words)
    with open(out_file, 'w') as f:
        f.write(' '.join(contents))

    return contents

def xml2text(in_dir, out_dir, white_list=None):
    # it will be fast if white_list is a dict instead of a list
    files = get_all_files(in_dir, recursive=False)
    count = 0
    for filename in files:
        if white_list and not os.path.basename(filename) in white_list:
            continue
        try:
            with open(filename, 'r') as fp:
                text = fp.read().lower()
                extract_contents(text, os.path.join(out_dir, os.path.basename(filename)))
                count += 1
        except Exception as e:
            raise e
        if count % 500 == 0:
            print 'processed %s' % count
    print 'processed %s docs, discarded %s docs' % (count, len(files) - count)

def load_data(corpus_dir, test_split, seed=666, stem=True):
    '''Loads the Wiki10+ dataset.

    @Params
        corpus_dir : path to the corpus dir
        test_split : fraction of the dataset to be used as test data.
        seed : random seed for sample shuffling.
        stem : stem flag.
    '''
    # count the number of times a word appears in a doc
    corpus = {}
    files = get_all_files(corpus_dir, False)
    cached_stop_words = []
    # cached_stop_words = init_stopwords()
    count = 0
    for filename in files:
        try:
            with open(filename, 'r') as fp:
                text = fp.read().lower()
                # remove punctuations, stopwords and *unnecessary digits*, stemming
                words = tiny_tokenize(text, stem, cached_stop_words)
                corpus[os.path.basename(filename)] = dict(Counter(words)) # doc-word frequency
                count += 1
        except Exception as e:
            raise e
        if count % 500 == 0:
            print count

    corpus = corpus.items()
    np.random.seed(seed)
    np.random.shuffle(corpus)

    n_docs = len(corpus)
    train_data = dict(corpus[:-int(n_docs * test_split)])
    test_data = dict(corpus[-int(n_docs * test_split):])

    return train_data, test_data

def construct_train_test_corpus(corpus_dir, test_split, output, threshold=10, topn=2000):
    train_data, test_data = load_data(corpus_dir, test_split)
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

def extract_labels(docs, labels, output):
    # it will be fast if docs is a dict instead of a list
    doc_labels = {}
    for name in docs:
        doc_labels[name] = labels[name]
    dump_json(doc_labels, output)
    import pdb;pdb.set_trace()
    return doc_labels
