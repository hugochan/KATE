'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse

from autoencoder.testing.visualize import visualize_pca_2d, plot_tsne
from autoencoder.utils.io_utils import load_json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_codes_file', type=str, help='path to the input corpus file')
    parser.add_argument('doc_labels_file', type=str, help='path to the output doc codes file')
    args = parser.parse_args()

    # 20news
    # visualize_pca_2d(load_json(args.doc_codes_file), load_json(args.doc_labels_file), ["rec.sport.hockey", "comp.graphics", "sci.crypt",
    #                                                             "soc.religion.christian", "talk.politics.mideast",
    #                                                             "talk.politics.guns"])

    plot_tsne(load_json(args.doc_codes_file), load_json(args.doc_labels_file), ["rec.sport.hockey", "comp.graphics", "sci.crypt",
                                                                "soc.religion.christian", "talk.politics.mideast",
                                                                "talk.politics.guns"])

    # # 8k
    # plot_tsne(load_json(sys.argv[1]), load_json(sys.argv[2]), [0, 1])
    # plot_tsne(load_json(sys.argv[1]), load_json(sys.argv[2]), ['1143155', '889936', '1362719', '700733', '730708'])
    # visualize_pca_2d(load_json(sys.argv[1]), load_json(sys.argv[2]), ['2006', '2008', '2010', '2012'])

