'''
Created on Jan, 2017

@author: hugo

'''
from __future__ import absolute_import
import os
import sys
import argparse
import numpy as np

from autoencoder.preprocessing.preprocessing import load_corpus, doc2vec
from autoencoder.utils.io_utils import load_json, dump_pickle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus', type=str, help='path to the corpus file')
    parser.add_argument('labels', type=str, help='path to the labels file')
    parser.add_argument('-bs', '--batch_size', type=int, default=100, help='batch size (default 100)')
    parser.add_argument('out_dir', type=str, help='path to the output dir')
    args = parser.parse_args()

    corpus = load_corpus(args.corpus)
    doc_labels = load_json(args.labels)
    vocab, docs = corpus['vocab'], corpus['docs']
    n_vocab = len(vocab)
    doc_names = docs.keys()
    X_docs = [doc2vec(x, n_vocab) for x in docs.values()]

    out_dir = args.out_dir
    # attributes
    attrs = zip(*sorted(vocab.iteritems(), key=lambda d:[1]))[0]
    dump_pickle(attrs, os.path.join(out_dir, 'attributes.p'))

    # batches
    bs = args.batch_size
    batches = [bs * (x + 1) for x in range(int(len(docs) / bs) - 1)]
    batches.append(len(docs))
    dump_pickle(batches, os.path.join(out_dir, 'batches.p'))

    # bow_batch_x
    for i in range(len(batches)):
        dump_pickle(X_docs[batches[i - 1] if i > 0 else 0: batches[i]], os.path.join(out_dir, 'bow_batch_%s.p' % batches[i]))

    # # docs_names_batch_x
    # for i in range(len(batches)):
    #     dump_pickle(doc_names[batches[i - 1] if i > 0 else 0: batches[i]], os.path.join(out_dir, 'docs_names_batch_%s.p' % batches[i]))

    # class_indices_batch_x
    for i in range(len(batches)):
        data = [doc_labels[doc_names[idx]] for idx in range(batches[i - 1] if i > 0 else 0, batches[i])]
        dump_pickle(data, os.path.join(out_dir, 'class_indices_batch_%s.p' % batches[i]))

    import pdb;pdb.set_trace()

if __name__ == "__main__":
    main()
