'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def plot_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    classes_to_visual = list(set(classes_to_visual))
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual:
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=c)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title("tsne")
    plt.savefig(save_file)
    plt.show()


def visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    classes_to_visual = list(set(classes_to_visual))
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 1, 2

    for c in classes_to_visual:
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=c)
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()

def DBN_plot_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual.keys(), range(C)))

    codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual.keys():
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title("tsne")
    plt.savefig(save_file)
    plt.show()

def DBN_visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_ids = dict(zip(classes_to_visual.keys(), range(C)))

    codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 0, 1

    for c in classes_to_visual.keys():
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()


def reuters_visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """

    # markers = ["p", "s", "h", "H", "+", "x", "D"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers

    class_names = classes_to_visual.keys()
    class_ids = dict(zip(class_names, range(C)))
    class_names = set(class_names)
    codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if class_names.intersection(set(doc_labels[doc]))])

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 1, 2

    for c in class_names:
        idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()

def get_indices(labels, c):
    idx = np.zeros(len(labels), dtype=bool)
    for i in range(len(labels)):
        tmp = [labels[i]] if not isinstance(labels[i], list) else labels[i]
        if c in tmp:
            idx[i] = True
    return idx
