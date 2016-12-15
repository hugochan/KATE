'''
Created on Nov, 2016

@author: hugo

'''

import os
import sys
import json
import numpy as np
from scipy import sparse
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


word_tokenizer = RegexpTokenizer(r'[a-zA-Z]+') # match only alphabet characters

def load_stopwords(file):
    stop_words = []
    try:
        with open(file, 'r') as f:
            for line in f:
                stop_words.append(line.strip('\n '))
    except Exception as e:
        raise e

    return stop_words

try:
    cached_stop_words = load_stopwords('patterns/stopwords.txt')
    print 'loaded patterns/stopwords.txt'
except:
    from nltk.corpus import stopwords
    cached_stop_words = stopwords.words("english")
    print 'loaded nltk.corpus.stopwords'

# def get_val_by_ids(lis, ids):
#     ret = []
#     try:
#         for i in ids:
#             ret.append(lis[i])
#     except Exception as e:
#         raise e
#     else:
#         return ret

def save_json(data, file):
    try:
        with open(file, 'w') as datafile:
            json.dump(data, datafile)
    except Exception as e:
        raise e

def load_json(file):
    try:
        with open(file, 'r') as datafile:
            data = json.load(datafile)
    except Exception as e:
        raise e

    return data

def get_all_files(corpus_path, recursive=False):
    if recursive:
        return [os.path.join(root, file) for root, dirnames, filenames in os.walk(corpus_path) for file in filenames if not file.startswith('.')]
    else:
        return [os.path.join(corpus_path, filename) for filename in os.listdir(corpus_path) if os.path.isfile(os.path.join(corpus_path, filename)) and not filename.startswith('.')]

def save_corpus(out_corpus, doc_word_freq, vocab_dict, word_freq):
    docs = {}
    for filename, val in doc_word_freq.iteritems():
        word_count = {}
        for word, freq in val.iteritems():
            try:
                word_count[vocab_dict[word]] = freq
            except: # word is not in vocab, i.e., this word should be filtered out
                pass
        docs[filename] = word_count
    corpus = {'docs': docs, 'vocab': vocab_dict, 'word_freq': word_freq}
    save_json(corpus, out_corpus)


def load_data(corpus_path, recursive=False):
    word_freq = defaultdict(lambda: 0) # count the number of times a word appears in a corpus
    doc_word_freq = defaultdict(dict) # count the number of times a word appears in a doc
    files = get_all_files(corpus_path, recursive)
    import pdb;pdb.set_trace()
    for filename in files:
        try:
            with open(filename, 'r') as fp:
                text = fp.read().lower()
                words = word_tokenizer.tokenize(text)
                words = [word for word in words if word not in cached_stop_words]

                for i in range(len(words)):
                    # doc-word frequency
                    basename = os.path.basename(filename)
                    try:
                        doc_word_freq[basename][words[i]] += 1
                    except:
                        doc_word_freq[basename][words[i]] = 1
                    # word frequency
                    word_freq[words[i]] += 1
        except Exception as e:
            print e
            sys.exit()

    return word_freq, doc_word_freq

def get_vocab_dict(word_freq, threshold=5, topn=None):
    idx = 0
    vocab_dict = {}
    if topn:
        word_freq = dict(sorted(word_freq.items(), key=lambda d:d[1], reverse=True)[:topn])
    for word, freq in word_freq.iteritems():
        if freq < threshold:
            continue
        vocab_dict[word] = idx
        idx += 1
    return vocab_dict

# def get_low_freq_words(word_freq, threshold=5):
#     return [word for word, freq in word_freq.iteritems() if freq < threshold]


def construct_corpus(corpus_path, out_corpus, threshold=5, recursive=False):
    word_freq, doc_word_freq = load_data(corpus_path, recursive)
    print 'finished loading'
    vocab_dict = get_vocab_dict(word_freq, threshold=threshold, topn=None)
    new_word_freq = dict([(word, freq) for word, freq in word_freq.items() if word in vocab_dict])
    save_corpus(out_corpus, doc_word_freq, vocab_dict, new_word_freq)

def load_corpus(corpus_path):
    corpus = load_json(corpus_path)

    return corpus

def vecnorm(vec, norm, epsilon=1e-3):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    Output will be in the same format as input (i.e., gensim vector=>gensim vector,
    or np array=>np array, scipy.sparse=>scipy.sparse).
    """
    if norm not in ('prob', 'max1', 'logmax1'):
        raise ValueError("'%s' is not a supported norm. Currently supported norms are 'prob' and 'max1'." % norm)
    # if sparse.issparse(vec):
    #     vec = vec.tocsr()
    #     if norm == 'prob':
    #         veclen = np.sum(np.abs(vec.data)) + epsilon * len(vec.data)
    #     if norm == 'max1':
    #         veclen = np.max(vec.data) + epsilon
    #     if veclen > 0.0:
    #         return (vec + epsilon) / veclen
    #     else:
    #         return vec

    if isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=float)
        if norm == 'prob':
            veclen = np.sum(np.abs(vec)) + epsilon * len(vec) # smoothing
        elif norm == 'max1':
            veclen = np.max(vec) + epsilon
        elif norm == 'logmax1':
            vec = np.log10(1. + vec)
            veclen = np.max(vec) + epsilon
        if veclen > 0.0:
            return (vec + epsilon) / veclen
        else:
            return vec

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
        weights[idx] = word_freq[word]

    weights = np.clip(weights / max_, 0., 1.)

    return np.power(weights, ratio)

def vocab_weights_tfidf(vocab_dict, word_freq, docs, max_=100., ratio=.75):
    dim = len(vocab_dict)
    tf_vec = np.zeros((dim, 1))
    for word, idx in vocab_dict.items():
        tf_vec[idx] = 1. + np.log10(word_freq[word]) # log normalization

    idf_vec = idf(docs, dim)
    tfidf_vec = tf_vec * idf_vec

    tfidf_vec = np.clip(tfidf_vec, 0., 4.)
    return np.power(tfidf_vec, ratio)

def revdict(d):
    """
    Reverse a dictionary mapping.
    When two keys map to the same value, only one of them will be kept in the
    result (which one is kept is arbitrary).
    """
    return dict((v, k) for (k, v) in d.iteritems())

def l1norm(x):
    return x / sum([np.abs(y) for y in x])


def corrupted_matrix(X, corruption_ratio=0.5):
    X_noisy = X + corruption_ratio * np.random.normal(loc=0.0, scale=1.0, size=X.shape)
    X_noisy = np.clip(X_noisy, 0., 1.)

    return X_noisy

def init_weights(topic_vocab_dist, vocab_dict, epsilon=1e-5):
    weights = np.zeros((len(vocab_dict), len(topic_vocab_dist)))
    for i in range(len(topic_vocab_dist)):
        for k, v in topic_vocab_dist[i]:
            weights[vocab_dict[k]][i] = 1. + epsilon

    return weights

def init_weights2(topic_vocab, vocab_dict, epsilon=1e-5):
    weights = np.zeros((len(vocab_dict), len(topic_vocab)))
    for i in range(len(topic_vocab)):
        for vocab in topic_vocab[i]:
            weights[vocab_dict[vocab]][i] = 1. / len(topic_vocab[i]) + epsilon

    return weights


def get_20news_doc_labels(corpus_path):
    doc_labels = defaultdict(list)
    files = get_all_files(corpus_path, True)
    for filename in files:
        label, name = filename.split('/')[-2:]
        doc_labels[name].append(label)
    # dirs = os.listdir(corpus_path)

    # doc_labels = {}
    # for each in dirs:
    #     if not os.path.isdir(each):
    #         continue
    #     docs = os.listdir(os.path.join(corpus_path, each))
    #     for doc in docs:
    #         doc_labels[doc] = each

    return doc_labels


def get_8k_doc_labels(doc_names):
    doc_labels = {}
    for doc in doc_names:
        doc_labels[doc] = doc.split('-')[-1].replace('.txt', '')

    return doc_labels

if __name__ == "__main__":
    usage = 'python utils.py [corpus_path] [out_corpus]'
    try:
        corpus_path = sys.argv[1]
        out_corpus = sys.argv[2]
    except:
        print usage
        sys.exit()

    construct_corpus(corpus_path, out_corpus, recursive=True)
    doc_labels = get_20news_doc_labels(corpus_path)
    # corpus = load_corpus(corpus_path)
    # doc_labels = get_8k_doc_labels(corpus['docs'].keys())
    # save_json(doc_labels, out_corpus)

    # bank_fyear = load_json(corpus_path)
    # doc_labels = load_json(out_corpus)
    # for doc, bname in doc_labels.items():    doc_fails[doc]=0 if bank_fyear[bname] == 'NA' else 1
