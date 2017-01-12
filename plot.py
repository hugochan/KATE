'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse

from autoencoder.testing.visualize import visualize_pca_2d, plot_tsne
from autoencoder.utils.io_utils import load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_codes_file', type=str, help='path to the input corpus file')
    parser.add_argument('doc_labels_file', type=str, help='path to the output doc codes file')
    parser.add_argument('cmd', choices=['pca', 'tsne'], help='plot cmd')
    parser.add_argument('-o', '--output', type=str, default='out.png', help='path to the output file')
    args = parser.parse_args()

    cmd = args.cmd.lower()
    classes_to_visual = ["rec.sport.hockey", "comp.graphics", "sci.crypt", \
                            "soc.religion.christian", "talk.politics.mideast", \
                            "talk.politics.guns"]

    # classes_to_visual = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware',
                        # 'comp.sys.mac.hardware', 'comp.windows.x']
    # classes_to_visual = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']

    # 20news
    if cmd == 'pca':
        visualize_pca_2d(load_json(args.doc_codes_file), load_json(args.doc_labels_file), classes_to_visual, args.output)
    elif cmd == 'tsne':
        plot_tsne(load_json(args.doc_codes_file), load_json(args.doc_labels_file), classes_to_visual, args.output)
    # # 8k
    # plot_tsne(load_json(sys.argv[1]), load_json(sys.argv[2]), [0, 1])
    # plot_tsne(load_json(sys.argv[1]), load_json(sys.argv[2]), ['1143155', '889936', '1362719', '700733', '730708'])
    # visualize_pca_2d(load_json(sys.argv[1]), load_json(sys.argv[2]), ['2006', '2008', '2010', '2012'])

if __name__ == '__main__':
    main()
