'''
Created on Apr, 2017

@author: hugo

'''

import numpy as np

def calc_ranks(x):
    """Given a list of items, return a list(in ndarray type) of ranks.
    """
    n = len(x)
    index = list(zip(*sorted(list(enumerate(x)), key=lambda d:d[1], reverse=True))[0])
    rank = np.zeros(n)
    rank[index] = range(1, n + 1)
    return rank

def rank_bank_topic(bank_doc_map, doc_topic_dist):
    """Rank topics for banks
    """
    bank_topic_ranks = {}
    for each_bank in bank_doc_map:
        rank = []
        for each_doc in bank_doc_map[each_bank]:
            rank.append(calc_ranks(doc_topic_dist[each_doc]))
        rank = np.r_[rank]
        # compute ranking score
        bank_topic_ranks[each_bank] = np.mean(1. / rank, axis=0)
    return bank_topic_ranks

if __name__ == '__main__':
    n = 10
    bank_doc_map = {'bank_0': ['doc_0', 'doc_1'], 'bank_1': ['doc_2', 'doc_3', 'doc_4']}
    doc_topic_dist = dict([('doc_%s' % i, np.random.randn(n)) for i in range(5)])
    rank = rank_bank_topic(bank_doc_map, doc_topic_dist)
