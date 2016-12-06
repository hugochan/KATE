'''
Created on Dec, 2016

@author: hugo

'''

import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from utils import *


def visualize_pca_2d(doc_codes, doc_labels, classes_to_visual):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual, range(C)))

    codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    X = np.r_[list(codes)]
    X = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual:
        idx = np.array(labels) == c
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=c)
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('1st PC')
    plt.ylabel('2nd PC')
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig('pca_2d.png')
    plt.show()

if __name__ == '__main__':
    visualize_pca_2d(load_json(sys.argv[1]), load_json(sys.argv[2]), ["rec.sport.hockey", "comp.graphics", "sci.crypt",
                                                                "soc.religion.christian", "talk.politics.mideast",
                                                                "talk.politics.guns"])

