'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse

from autoencoder.testing.visualize import visualize_pca_2d, visualize_pca_3d, plot_tsne, plot_tsne_3d
from autoencoder.utils.io_utils import load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_codes_file', type=str, help='path to the input corpus file')
    parser.add_argument('doc_labels_file', type=str, help='path to the output doc codes file')
    parser.add_argument('cmd', choices=['pca', 'tsne'], help='plot cmd')
    parser.add_argument('-o', '--output', type=str, default='out.png', help='path to the output file')
    args = parser.parse_args()

    cmd = args.cmd.lower()

    # classes_to_visual = ["rec.sport.hockey", "comp.graphics", "sci.crypt", \
    #                         "soc.religion.christian", "talk.politics.mideast", \
    #                         "talk.politics.guns"]

    # classes_to_visual = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                        # 'comp.sys.mac.hardware', 'comp.windows.x']
    # classes_to_visual = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    doc_codes = load_json(args.doc_codes_file)
    doc_labels = load_json(args.doc_labels_file)

    # 20news
    # if cmd == 'pca':
    #     visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, args.output)
    # elif cmd == 'tsne':
    #     plot_tsne(doc_codes, doc_labels, classes_to_visual, args.output)

    # 8k
    classes_to_visual = ["1", "2", "3", "4", "5", "7", "8"]
    for k in doc_labels:
        doc_labels[k] = doc_labels[k].split('.')[0]

    # 10k
    # classes_to_visual = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15"]
    # for k in doc_labels:
    #     doc_labels[k] = ''.join([y for y in list(doc_labels[k]) if y.isdigit()])

    if cmd == 'pca':
        visualize_pca_3d(doc_codes, doc_labels, classes_to_visual, args.output)
    elif cmd == 'tsne':
        plot_tsne_3d(doc_codes, doc_labels, classes_to_visual, args.output)

if __name__ == '__main__':
    main()
