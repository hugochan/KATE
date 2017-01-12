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
    legend = plt.legend(loc='upper right', shadow=True)
    # plt.title("tsne")
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
    # plt.title('Projected on the PCA components')
    # plt.xlabel('PC %s' % x_pc)
    # plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper right', shadow=True)
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

    x_pc, y_pc = 1, 2

    for c in classes_to_visual.keys():
        idx = np.array(labels) == c
        # idx = get_indices(labels, c)
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
        # plt.legend(c)
    plt.title('Projected on the first 2 PCs')
    plt.xlabel('PC %s' % x_pc)
    plt.ylabel('PC %s' % y_pc)
    # legend = plt.legend(loc='upper center', shadow=True)
    plt.savefig(save_file)
    plt.show()


def reuters_visualize_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
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
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    plt.figure(figsize=(10, 10), facecolor='white')

    for c in classes_to_visual.keys():
        idx = get_indices(labels, c)
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=0.6, marker=markers[class_ids[c]],
                        markersize=6, label=classes_to_visual[c])
    legend = plt.legend(loc='upper center', shadow=True)
    plt.title("tsne")
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
    codes, labels = zip(*[(code, class_names.intersection(set(doc_labels[doc]))) for doc, code in doc_codes.items() if len(class_names.intersection(set(doc_labels[doc]))) == 1])
    # codes = []
    # labels = []
    # for doc, code in doc_codes.items():
    #     y = set(doc_labels[doc])
    #     x = list(class_names.intersection(y))
    #     if x:
    #         codes.append(code)
    #         labels.append(x[0])
    # x = 0
    # pairs = []
    # for each in labels:
    #     if len(class_names.intersection(set(each))) > 1:
    #         x += 1
    #         pairs.append(class_names.intersection(set(each)))
    # print x


    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    plt.figure(figsize=(10, 10), facecolor='white')

    x_pc, y_pc = 0, 1

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
        tmp = [labels[i]] if not isinstance(labels[i], (list, set)) else labels[i]
        if c in tmp:
            idx[i] = True
    return idx

def plot_info_retrieval(precisions, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]

    i = 0
    for model_name, val in precisions.iteritems():
        fr, pr = zip(*val)
        plt.plot(fr, pr, linestyle='-', alpha=0.6, marker=markers[i],
                        markersize=6, label=model_name)
        i += 1
        # plt.legend(model_name)
    # plt.title('Projected on the PCA components')
    plt.xlabel('Fraction of Retrieved Documents (%)')
    plt.ylabel('Precision (%)')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

if __name__ == '__main__':
    import sys
    precisions = {
        'LDA': [(0.001, 0.4426326461642425), (0.002, 0.41994158258098285), (0.005, 0.38933170851984034), (0.01, 0.36552361840219), (0.02, 0.33586314908895326), (0.05, 0.2728201561244453), (0.1, 0.2070775654290608), (0.2, 0.14147130803649285), (0.5, 0.07701982354516679), (1.0, 0.05050301672031405)],
        'DBN': [(0.001, 0.535038381692656), (0.002, 0.5077608265340592), (0.005, 0.465912108337758), (0.01, 0.4264154357337848), (0.02, 0.37657322856108594), (0.05, 0.29198182151435376), (0.1, 0.2197600288870639), (0.2, 0.15325609847145583), (0.5, 0.08605947016611057), (1.0, 0.05050301672031403)],
        'DocNADE': [(0.001, 0.5718148022980761), (0.002, 0.5435414956790445), (0.005, 0.5074230900538642), (0.01, 0.4746133312027964), (0.02, 0.43102761550716634), (0.05, 0.3383512940656766), (0.1, 0.25088957318799715), (0.2, 0.16893256617330504), (0.5, 0.0898931631614369), (1.0, 0.05050301672031405)],
        'NVDM': [(0.001, 0.05129628735576586), (0.002, 0.0513143919277744), (0.005, 0.0513784045216623), (0.01, 0.05057947447821584), (0.02, 0.04999729766565648), (0.05, 0.05015274063700316), (0.1, 0.050297158296132634), (0.2, 0.05053768818029844), (0.5, 0.050492220758456205), (1.0, 0.05050301672031404)],
        'AvgWord2Vec': [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)],
        'AE': [(0.001, 0.22827451359049142), (0.002, 0.18935571863080838), (0.005, 0.14794495865260598), (0.01, 0.12336861250406352), (0.02, 0.10404868431566056), (0.05, 0.08489066120247557), (0.1, 0.07413015988839661), (0.2, 0.06568426232571814), (0.5, 0.0563362391994616), (1.0, 0.05050301672031405)],
        'VAE': [(0.001, 0.1625428474870794), (0.002, 0.129049389272433), (0.005, 0.09866000303467093), (0.01, 0.0835667523580887), (0.02, 0.07314337881088674), (0.05, 0.06373537802133396), (0.1, 0.05873448646810881), (0.2, 0.05491787941153268), (0.5, 0.05159197722972036), (1.0, 0.05050301672031405)],
        'KSAE': [(0.001, 0.23964418481146235), (0.002, 0.20264447448462075), (0.005, 0.16342178135194532), (0.01, 0.1395696943777457), (0.02, 0.12070563824438621), (0.05, 0.09967618984957147), (0.1, 0.08657995851945419), (0.2, 0.07516400405132624), (0.5, 0.061121338068411066), (1.0, 0.05050301672031405)],
        'KCAE': [(0.001, 0.5469994689325542), (0.002, 0.5114722637956814), (0.005, 0.4632615127835514), (0.01, 0.42367432876364286), (0.02, 0.3765967271206238), (0.05, 0.2956349075801771), (0.1, 0.22285792231953094), (0.2, 0.15466652626952637), (0.5, 0.0866775624520581), (1.0, 0.05050301672031405)]}
    plot_info_retrieval(precisions, sys.argv[1])
