'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import
import argparse

from autoencoder.testing.visualize import reuters_visualize_pca_2d, reuters_visualize_tsne
from autoencoder.utils.io_utils import load_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('doc_codes_file', type=str, help='path to the input corpus file')
    parser.add_argument('doc_labels_file', type=str, help='path to the output doc codes file')
    parser.add_argument('cmd', choices=['pca', 'tsne'], help='plot cmd')
    parser.add_argument('-o', '--output', type=str, default='out.png', help='path to the output file')
    args = parser.parse_args()

    cmd = args.cmd.lower()
    classes_to_visual = {'ECAT': 'ECONOMICS', 'MCAT': 'MARKETS',
                        'CCAT': 'CORPORATE/INDUSTRIAL', 'GCAT': 'GOVERNMENT/SOCIAL'}

    if cmd == 'pca':
        reuters_visualize_pca_2d(load_json(args.doc_codes_file), load_json(args.doc_labels_file), classes_to_visual, args.output)
    elif cmd == 'tsne':
        reuters_visualize_tsne(load_json(args.doc_codes_file), load_json(args.doc_labels_file), classes_to_visual, args.output)


if __name__ == '__main__':
    main()
