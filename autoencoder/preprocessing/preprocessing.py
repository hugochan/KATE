'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import os
import re
import string
import codecs
import numpy as np
from collections import defaultdict
from nltk.tokenize import wordpunct_tokenize
from nltk.stem.porter import PorterStemmer as EnglishStemmer
# from nltk.tokenize import RegexpTokenizer

from ..utils.io_utils import dump_json, load_json, write_file


def load_stopwords(file):
    stop_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                stop_words.append(line.strip('\n '))
    except Exception as e:
        raise e

    return stop_words

def init_stopwords():
    try:
        stopword_path = 'patterns/english_stopwords.txt'
        cached_stop_words = load_stopwords(os.path.join(os.path.split(__file__)[0], stopword_path))
        print 'Loaded %s' % stopword_path
    except:
        from nltk.corpus import stopwords
        cached_stop_words = stopwords.words("english")
        print 'Loaded nltk.corpus.stopwords'

    return cached_stop_words

def tiny_tokenize(text, stem=False, stop_words=[]):
    words = []
    for token in wordpunct_tokenize(re.sub('[%s]' % re.escape(string.punctuation), ' ', \
            text.decode(encoding='UTF-8', errors='ignore'))):
        if not token.isdigit() and not token in stop_words:
            if stem:
                try:
                    w = EnglishStemmer().stem(token)
                except Exception as e:
                    w = token
            else:
                w = token
            words.append(w)

    return words

    # return [EnglishStemmer().stem(token) if stem else token for token in wordpunct_tokenize(
    #                     re.sub('[%s]' % re.escape(string.punctuation), ' ', text.decode(encoding='UTF-8', errors='ignore'))) if
    #                     not token.isdigit() and not token in stop_words]

def tiny_tokenize_xml(text, stem=False, stop_words=[]):
    return [EnglishStemmer().stem(token) if stem else token for token in wordpunct_tokenize(
                        re.sub('[%s]' % re.escape(string.punctuation), ' ', text.encode(encoding='ascii', errors='ignore'))) if
                        not token.isdigit() and not token in stop_words]

def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if os.path.isfile(os.path.join(root, file)) and not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]

def count_words(docs):
    # count the number of times a word appears in a corpus
    word_freq = defaultdict(lambda: 0)
    for each in docs:
        for word, val in each.iteritems():
            word_freq[word] += val

    return word_freq

def load_data(corpus_path, recursive=False, stem=False, stop_words=False):
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc
    files = get_all_files(corpus_path, recursive)

    # word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # match only alphabet characters
    # cached_stop_words = init_stopwords()
    cached_stop_words = init_stopwords() if stop_words else []

    for filename in files:
        try:
            # with open(filename, 'r') as fp:
            with codecs.open(filename, 'r', encoding='UTF-8', errors='ignore') as fp:
                text = fp.read().lower()
                # words = [word for word in word_tokenizer.tokenize(text) if word not in cached_stop_words]
                # remove punctuations, stopwords and *unnecessary digits*, stemming
                words = tiny_tokenize(text.decode('utf-8'), stem, cached_stop_words)

                # doc_name = os.path.basename(filename)
                parent_name, child_name = os.path.split(filename)
                doc_name = os.path.split(parent_name)[-1] + '_' + child_name
                for i in range(len(words)):
                    # doc-word frequency
                    try:
                        doc_word_freq[doc_name][words[i]] += 1
                    except:
                        doc_word_freq[doc_name][words[i]] = 1
                    # word frequency
                    word_freq[words[i]] += 1
        except Exception as e:
            raise e

    return word_freq, doc_word_freq

def construct_corpus(corpus_path, training_phase, vocab_dict=None, threshold=5, topn=None, recursive=False):
    if not (training_phase or isinstance(vocab_dict, dict)):
        raise ValueError('vocab_dict must be provided if training_phase is set False')

    word_freq, doc_word_freq = load_data(corpus_path, recursive)
    if training_phase:
        vocab_dict = build_vocab(word_freq, threshold=threshold, topn=topn)

    docs = generate_bow(doc_word_freq, vocab_dict)
    new_word_freq = dict([(vocab_dict[word], freq) for word, freq in word_freq.iteritems() if word in vocab_dict])

    return docs, vocab_dict, new_word_freq

def load_corpus(corpus_path):
    corpus = load_json(corpus_path)

    return corpus

def generate_bow(doc_word_freq, vocab_dict):
    docs = {}
    for key, val in doc_word_freq.iteritems():
        word_count = {}
        for word, freq in val.iteritems():
            try:
                word_count[vocab_dict[word]] = freq
            except: # word is not in vocab, i.e., this word should be discarded
                continue
        docs[key] = word_count

    return docs

def build_vocab(word_freq, threshold=5, topn=None, start_idx=0):
    """
    threshold only take effects when topn is None.
    words are indexed by overall frequency in the dataset.
    """
    word_freq = sorted(word_freq.iteritems(), key=lambda d:d[1], reverse=True)
    if topn:
        word_freq = zip(*word_freq[:topn])[0]
        vocab_dict = dict(zip(word_freq, range(start_idx, len(word_freq) + start_idx)))
    else:
        idx = start_idx
        vocab_dict = {}
        for word, freq in word_freq:
            if freq < threshold:
                return vocab_dict
            vocab_dict[word] = idx
            idx += 1
    return vocab_dict

def construct_train_test_corpus(train_path, test_path, output, threshold=5, topn=None):
    train_docs, vocab_dict, train_word_freq = construct_corpus(train_path, True, threshold=threshold, topn=topn, recursive=True)
    train_corpus = {'docs': train_docs, 'vocab': vocab_dict, 'word_freq': train_word_freq}
    dump_json(train_corpus, os.path.join(output, 'train.corpus'))
    print 'Generated training corpus'

    test_docs, _, _ = construct_corpus(test_path, False, vocab_dict=vocab_dict, recursive=True)
    test_corpus = {'docs': test_docs, 'vocab': vocab_dict}
    dump_json(test_corpus, os.path.join(output, 'test.corpus'))
    print 'Generated test corpus'

    return train_corpus, test_corpus

def corpus2libsvm(docs, doc_labels, output):
    '''Convert the corpus format to libsvm format.
    '''
    data = []
    names = []
    for key, val in docs.iteritems():
        # label = doc_labels[key]
        label = 0
        line = label if isinstance(label, list) else [str(label)] + ["%s:%s" % (int(x) + 1, y) for x, y in val.iteritems()]
        data.append(line)
        names.append(key)
    write_file(data, output)
    write_file(names, output + '.fnames')
    return data, names

def doc2vec(doc, dim):
    vec = np.zeros(dim)
    for idx, val in doc.items():
        vec[int(idx)] = val

    return vec

def idf(docs, dim):
    vec = np.zeros((dim, 1))
    for each_doc in docs:
        for idx in each_doc.keys():
            vec[int(idx)] += 1

    return np.log10(1. + len(docs) / vec)

def vocab_weights(vocab_dict, word_freq, max_=100., ratio=.75):
    weights = np.zeros((len(vocab_dict), 1))

    for word, idx in vocab_dict.items():
        weights[idx] = word_freq[str(idx)]
    weights = np.clip(weights / max_, 0., 1.)

    return np.power(weights, ratio)

def vocab_weights_tfidf(vocab_dict, word_freq, docs, max_=100., ratio=.75):
    dim = len(vocab_dict)
    tf_vec = np.zeros((dim, 1))
    for word, idx in vocab_dict.items():
        tf_vec[idx] = 1. + np.log10(word_freq[idx]) # log normalization

    idf_vec = idf(docs, dim)
    tfidf_vec = tf_vec * idf_vec

    tfidf_vec = np.clip(tfidf_vec, 0., 4.)
    return np.power(tfidf_vec, ratio)

# # Init weights with topic modeling results
# def init_weights(topic_vocab_dist, vocab_dict, epsilon=1e-5):
#     weights = np.zeros((len(vocab_dict), len(topic_vocab_dist)))
#     for i in range(len(topic_vocab_dist)):
#         for k, v in topic_vocab_dist[i]:
#             weights[vocab_dict[k]][i] = 1. + epsilon

#     return weights

# def init_weights2(topic_vocab, vocab_dict, epsilon=1e-5):
#     weights = np.zeros((len(vocab_dict), len(topic_vocab)))
#     for i in range(len(topic_vocab)):
#         for vocab in topic_vocab[i]:
#             weights[vocab_dict[vocab]][i] = 1. / len(topic_vocab[i]) + epsilon

#     return weights

def generate_20news_doc_labels(doc_names, output):
    doc_labels = {}
    for each in doc_names:
       label = each.split('_')[0]
       doc_labels[each] = label

    dump_json(doc_labels, output)

    return doc_labels

def generate_8k_doc_labels(doc_names, output):
    doc_labels = {}
    for each in doc_names:
       label = each.split('_')[-1].replace('.txt', '')
       doc_labels[each] = label

    dump_json(doc_labels, output)

    return doc_labels

def get_8k_doc_bnames(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[-1].replace('.txt', '')

    return doc_labels

def get_8k_doc_years(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[0]

    return doc_labels

def get_8k_doc_fails(doc_names, bank_fyear):
    doc_labels = {}
    for doc in doc_names:
        fyear = bank_fyear[doc.split('-')[-1].replace('.txt', '')]
        doc_labels[doc] = 1 if fyear != 'NA' and abs(int(doc.split('-')[0]) - int(fyear)) <= 1 else 0

    return doc_labels
