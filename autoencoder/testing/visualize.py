'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, MultipleLocator
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate


class neural_net_visualizer(object):
    def __init__(self):
        pass



def heatmap(data, save_file='heatmap.png'):
    ax = plt.figure().gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    plt.pcolor(data, cmap=plt.cm.jet)
    plt.savefig(save_file)
    # plt.show()

def word_cloud(word_embedding_matrix, vocab, s, save_file='scatter.png'):
    words = [(i, vocab[i]) for i in s]
    model = TSNE(n_components=2, random_state=0)
    #Note that the following line might use a good chunk of RAM
    tsne_embedding = model.fit_transform(word_embedding_matrix)
    words_vectors = tsne_embedding[np.array([item[1] for item in words])]

    plt.subplots_adjust(bottom = 0.1)
    plt.scatter(
        words_vectors[:, 0], words_vectors[:, 1], marker='o', cmap=plt.get_cmap('Spectral'))

    for label, x, y in zip(s, words_vectors[:, 0], words_vectors[:, 1]):
        plt.annotate(
            label,
            xy=(x, y), xytext=(-20, 20),
            textcoords='offset points', ha='right', va='bottom',
            fontsize=20,
            # bbox=dict(boxstyle='round,pad=1.', fc='yellow', alpha=0.5),
            arrowprops=dict(arrowstyle = '<-', connectionstyle='arc3,rad=0')
            )
    plt.show()
    # plt.savefig(save_file)

def plot_tsne(doc_codes, doc_labels, classes_to_visual, save_file):
    # markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    plt.rc('legend',**{'fontsize':30})
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
        plt.plot(X[idx, 0], X[idx, 1], linestyle='None', alpha=1, marker=markers[class_ids[c]],
                        markersize=10, label=c)
    legend = plt.legend(loc='upper right', shadow=True)
    # plt.title("tsne")
    # plt.savefig(save_file)
    plt.savefig(save_file, format='eps', dpi=2000)
    plt.show()


def plot_tsne_3d(doc_codes, doc_labels, classes_to_visual, save_file, maker_size=None, opaque=None):
    markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    plt.rc('legend',**{'fontsize':20})
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers
    while True:
        if C <= len(colors):
            break
        colors += colors

    class_ids = dict(zip(classes_to_visual, range(C)))

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    tsne = TSNE(perplexity=30, n_components=3, init='pca', n_iter=5000)
    np.set_printoptions(suppress=True)
    X = tsne.fit_transform(X)

    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')

    # The problem is that the legend function don't support the type returned by a 3D scatter.
    # So you have to create a "dummy plot" with the same characteristics and put those in the legend.
    scatter_proxy = []
    for i in range(C):
        cls = classes_to_visual[i]
        idx = np.array(labels) == cls
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=colors[i], alpha=opaque[i] if opaque else 1, s=maker_size[i] if maker_size else 20, marker=markers[i], label=cls)
        scatter_proxy.append(mpl.lines.Line2D([0],[0], linestyle="none", c=colors[i], marker=markers[i], label=cls))
    ax.legend(scatter_proxy, classes_to_visual, numpoints=1)
    plt.savefig(save_file)
    plt.show()


def visualize_pca_2d(doc_codes, doc_labels, classes_to_visual, save_file):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """
    # markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    plt.rc('legend',**{'fontsize':28})
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
        plt.plot(X[idx, x_pc], X[idx, y_pc], linestyle='None', alpha=1, marker=markers[class_ids[c]],
                        markersize=10, label=c)
        # plt.legend(c)
    # plt.title('Projected on the PCA components')
    # plt.xlabel('PC %s' % x_pc)
    # plt.ylabel('PC %s' % y_pc)
    legend = plt.legend(loc='upper right', shadow=True)
    # plt.savefig(save_file)
    plt.savefig(save_file, format='eps', dpi=2000)
    plt.show()

def visualize_pca_3d(doc_codes, doc_labels, classes_to_visual, save_file, maker_size=None, opaque=None):
    """
        Visualize the input data on a 2D PCA plot. Depending on the number of components,
        the plot will contain an X amount of subplots.
        @param doc_codes:
        @param number_of_components: The number of principal components for the PCA plot.
    """
    markers = ["D", "p", "*", "s", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    plt.rc('legend',**{'fontsize':20})
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k']
    C = len(classes_to_visual)
    while True:
        if C <= len(markers):
            break
        markers += markers
    while True:
        if C <= len(colors):
            break
        colors += colors

    if isinstance(doc_codes, dict) and isinstance(doc_labels, dict):
        codes, labels = zip(*[(code, doc_labels[doc]) for doc, code in doc_codes.items() if doc_labels[doc] in classes_to_visual])
    else:
        codes, labels = doc_codes, doc_labels

    X = np.r_[list(codes)]
    X = PCA(n_components=3).fit_transform(X)
    fig = plt.figure(figsize=(10, 10), facecolor='white')
    ax = fig.add_subplot(111, projection='3d')
    x_pc, y_pc, z_pc = 0, 1, 2

    # The problem is that the legend function don't support the type returned by a 3D scatter.
    # So you have to create a "dummy plot" with the same characteristics and put those in the legend.
    scatter_proxy = []
    for i in range(C):
        cls = classes_to_visual[i]
        idx = np.array(labels) == cls
        ax.scatter(X[idx, x_pc], X[idx, y_pc], X[idx, z_pc], c=colors[i], alpha=opaque[i] if opaque else 1, s=maker_size[i] if maker_size else 20, marker=markers[i], label=cls)
        scatter_proxy.append(mpl.lines.Line2D([0],[0], linestyle="none", c=colors[i], marker=markers[i], label=cls))
    ax.legend(scatter_proxy, classes_to_visual, numpoints=1)
    # plt.title('Projected on the PCA components')
    ax.set_xlabel('%sst component' % (x_pc + 1), fontsize=14)
    ax.set_ylabel('%snd component' % (y_pc + 1), fontsize=14)
    ax.set_zlabel('%srd component' % (z_pc + 1), fontsize=14)
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
    # markers = ["|", "D", "8", "v", "^", ">", "h", "H", "s", "*", "p", "d", "<"]
    markers = ["D", "p", 's', "*", "d", "8", "^", "H", "v", ">", "<", "h", "|"]
    ticks = zip(*zip(*precisions)[1][0])[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions:
        fr, pr = zip(*val)
        plt.plot(new_x, pr, linestyle='-', alpha=0.7, marker=markers[i],
                        markersize=8, label=model_name)
        i += 1
        # plt.legend(model_name)
    plt.xlabel('Fraction of Retrieved Documents')
    plt.ylabel('Precision')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

def plot_info_retrieval_by_length(precisions, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "^", "x", "D"]
    ticks = zip(*zip(*precisions)[1][0])[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions:
        fr, pr = zip(*val)
        plt.plot(new_x, pr, linestyle='-', alpha=0.6, marker=markers[i],
                        markersize=6, label=model_name)
        i += 1
        # plt.legend(model_name)
    plt.xlabel('Document Sorted by Length')
    plt.ylabel('Precision (%)')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

def plot(x, y, x_label, y_label, save_file):
    ticks = x
    plt.xticks(range(len(ticks)), ticks, fontsize = 15)
    plt.yticks(fontsize = 15)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    plt.plot(new_x, y, linestyle='-', alpha=1.0, markersize=12, marker='p', color='b')
    plt.xlabel(x_label, fontsize=24)
    plt.ylabel(y_label, fontsize=20)
    plt.savefig(save_file)
    plt.show()

if __name__ == '__main__':
    import sys
    # 20news_retrieval_128D
    precisions = [
        ('VAE', [(0.001, 0.587348525080869), (0.002, 0.5651402500844888), (0.005, 0.5327151771489245), (0.01, 0.5014839340348453), (0.02, 0.4584269359288251), (0.05, 0.3658133556412997), (0.1, 0.2687164883998648), (0.2, 0.17739560251738207), (0.5, 0.09136516909151776), (1.0, 0.05050301672031405)]),
        ('DocNADE', [(0.001, 0.5718148022980761), (0.002, 0.5435414956790445), (0.005, 0.5074230900538642), (0.01, 0.4746133312027964), (0.02, 0.43102761550716634), (0.05, 0.3383512940656766), (0.1, 0.25088957318799715), (0.2, 0.16893256617330504), (0.5, 0.0898931631614369), (1.0, 0.05050301672031405)]),
        ('KATE', [(0.001, 0.5543982040264583), (0.002, 0.5213392555399969), (0.005, 0.4739445034519384), (0.01, 0.4347574243698827), (0.02, 0.3869114198299623), (0.05, 0.30403564261511706), (0.1, 0.2277761656366975), (0.2, 0.15699569840064684), (0.5, 0.08683891514289452), (1.0, 0.05050301672031405)]),
        ('DBN', [(0.001, 0.535038381692656), (0.002, 0.5077608265340592), (0.005, 0.465912108337758), (0.01, 0.4264154357337848), (0.02, 0.37657322856108594), (0.05, 0.29198182151435376), (0.1, 0.2197600288870639), (0.2, 0.15325609847145583), (0.5, 0.08605947016611057), (1.0, 0.05050301672031403)]),
        ('LDA',  [(0.001, 0.4867957321488915), (0.002, 0.46359774054941044), (0.005, 0.42999155982095444), (0.01, 0.40179481997753264), (0.02, 0.36320959775165296), (0.05, 0.2823678558504475), (0.1, 0.20784423242441585), (0.2, 0.14168207983103592), (0.5, 0.0800605531419018), (1.0, 0.05050301672031405)]),
        ('Word2Vec_pre', [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)]),
        ('Word2Vec', [(0.001, 0.3815960990682146), (0.002, 0.33990126973397794), (0.005, 0.28322016538957506), (0.01, 0.23994026666166862), (0.02, 0.1996232005978027), (0.05, 0.15086380704863955), (0.1, 0.11886672273161164), (0.2, 0.09250234660438485), (0.5, 0.06552700581695815), (1.0, 0.05050301672031405)]),
        ('CAE',  [(0.001, 0.25605899676530824), (0.002, 0.2178643846859439), (0.005, 0.17500331917153453), (0.01, 0.14827356083073284), (0.02, 0.12567969583465557), (0.05, 0.1013255537435644), (0.1, 0.08658477146491544), (0.2, 0.07424743141317958), (0.5, 0.059710587487142405), (1.0, 0.05050301672031405)]),
        ('KSAE', [(0.001, 0.23964418481146235), (0.002, 0.20264447448462075), (0.005, 0.16342178135194532), (0.01, 0.1395696943777457), (0.02, 0.12070563824438621), (0.05, 0.09967618984957147), (0.1, 0.08657995851945419), (0.2, 0.07516400405132624), (0.5, 0.061121338068411066), (1.0, 0.05050301672031405)]),
        ('AE', [(0.001, 0.22827451359049142), (0.002, 0.18935571863080838), (0.005, 0.14794495865260598), (0.01, 0.12336861250406352), (0.02, 0.10404868431566056), (0.05, 0.08489066120247557), (0.1, 0.07413015988839661), (0.2, 0.06568426232571814), (0.5, 0.0563362391994616), (1.0, 0.05050301672031405)]),
        ('DAE', [(0.001, 0.2095785255636476), (0.002, 0.17031574373581437), (0.005, 0.1300285448751998), (0.01, 0.10855864535504914), (0.02, 0.09279581161675608), (0.05, 0.07767683840981233), (0.1, 0.06946348101328252), (0.2, 0.06284691358720304), (0.5, 0.05536974244871757), (1.0, 0.05050301672031405)]),
        ('Doc2Vec', [(0.001, 0.16486023270409583), (0.002, 0.1494834162120367), (0.005, 0.12679472346559542), (0.01, 0.11052195000447207), (0.02, 0.0953665540302444), (0.05, 0.07877845088096704), (0.1, 0.06914265711214816), (0.2, 0.06190158066520009), (0.5, 0.05536713733618202), (1.0, 0.05050301672031404)]),
        ('NVDM', [(0.001, 0.05129628735576586), (0.002, 0.0513143919277744), (0.005, 0.0513784045216623), (0.01, 0.05057947447821584), (0.02, 0.04999729766565648), (0.05, 0.05015274063700316), (0.1, 0.050297158296132634), (0.2, 0.05053768818029844), (0.5, 0.050492220758456205), (1.0, 0.05050301672031404)])
        ]

    # 20news_retrieval_512D
    # precisions = {
    # 'LDA':  [(0.001, 0.4058682952734931), (0.002, 0.37058851928739733), (0.005, 0.3309972687959938), (0.01, 0.3016110612419513), (0.02, 0.2693117036925588), (0.05, 0.2161839279252353), (0.1, 0.1702690976502034), (0.2, 0.12488871530981528), (0.5, 0.07553462776603063), (1.0, 0.05050301672031405)],
    # 'DBN': [(0.001, 0.5553034326268522), (0.002, 0.5285147009124683), (0.005, 0.488347337076094), (0.01, 0.45024297510562405), (0.02, 0.4033891972422051), (0.05, 0.31771321417997606), (0.1, 0.23800250085341837), (0.2, 0.1643866217959289), (0.5, 0.08997211449990615), (1.0, 0.050503016720313966)],
    # 'DocNADE': [(0.001, 0.5771737556124188), (0.002, 0.5443682711340714), (0.005, 0.5036226386465347), (0.01, 0.47000408874935945), (0.02, 0.42806973432528334), (0.05, 0.3391154672218619), (0.1, 0.2523257091581672), (0.2, 0.16856842576301645), (0.5, 0.08886419064880063), (1.0, 0.05050301672031405)],
    # 'NVDM':  [(0.001, 0.051827354801331486), (0.002, 0.05166441365326164), (0.005, 0.05022143615810757), (0.01, 0.050504279087694), (0.02, 0.04984220717270456), (0.05, 0.050210312107871934), (0.1, 0.05031805352276878), (0.2, 0.050525479733275175), (0.5, 0.05048740951458409), (1.0, 0.05050301672031404)],
    # 'Word2Vec_pre': [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)],
    # 'Word2Vec': [(0.001, 0.3842755757253872), (0.002, 0.34131946120793055), (0.005, 0.28388162885972246), (0.01, 0.24101532576054588), (0.02, 0.20010492106833672), (0.05, 0.1509728403648974), (0.1, 0.118880457234514), (0.2, 0.09250264007666884), (0.5, 0.0655253160142321), (1.0, 0.05050301672031405)],
    # 'Doc2Vec':  [(0.001, 0.2199705498961938), (0.002, 0.19429826678896794), (0.005, 0.15887688718610055), (0.01, 0.13127352793274671), (0.02, 0.1067768670780567), (0.05, 0.08213522011101435), (0.1, 0.06901493797404573), (0.2, 0.05975776562880724), (0.5, 0.05340344575184013), (1.0, 0.050503016720314126)],
    # 'AE':  [(0.001, 0.25062762516293363), (0.002, 0.2055713802925667), (0.005, 0.15574975343297148), (0.01, 0.12843020222861343), (0.02, 0.10797588107849927), (0.05, 0.08867698410088286), (0.1, 0.07749710871105607), (0.2, 0.06824739056183722), (0.5, 0.057956525318736705), (1.0, 0.05050301672031405)],
    # 'DAE': [(0.001, 0.2441703278134424), (0.002, 0.19592767826968308), (0.005, 0.14740440785979805), (0.01, 0.12183063178228161), (0.02, 0.1024848551783867), (0.05, 0.08429144793424902), (0.1, 0.07442198872784735), (0.2, 0.06632051023795682), (0.5, 0.05714302612312982), (1.0, 0.05050301672031405)],
    # 'CAE':  [(0.001, 0.2564210882054672), (0.002, 0.20924660841017487), (0.005, 0.16076881496092835), (0.01, 0.1325765230591466), (0.02, 0.11132618820467256), (0.05, 0.09089712800606133), (0.1, 0.07922084751978385), (0.2, 0.06933241629113954), (0.5, 0.058349474860945535), (1.0, 0.05050301672031405)],
    # 'VAE':  [(0.001, 0.2864384685945949), (0.002, 0.22214913339448517), (0.005, 0.15730739321751025), (0.01, 0.12415463932062157), (0.02, 0.10121123325141194), (0.05, 0.08010235972535741), (0.1, 0.06876865603310822), (0.2, 0.06069940080002707), (0.5, 0.05329987492643488), (1.0, 0.05050301672031405)],
    # 'KSAE': [(0.001, 0.2766257905663044), (0.002, 0.23146091826389079), (0.005, 0.18327753964039104), (0.01, 0.15379102261032626), (0.02, 0.13065375342492558), (0.05, 0.10653729926356474), (0.1, 0.09147214149778002), (0.2, 0.07821235936221227), (0.5, 0.0621660351341905), (1.0, 0.05050301672031405)],
    # 'KATE': [(0.001, 0.5370057451841842), (0.002, 0.49623424902234925), (0.005, 0.4398091950534882), (0.01, 0.39517410082761617), (0.02, 0.3440230238886337), (0.05, 0.26452986431933806), (0.1, 0.19919513465212774), (0.2, 0.14045712651660616), (0.5, 0.08193839335997657), (1.0, 0.05050301672031405)]}

    # precisions = {
    # 'DocNADE': [(100, 0.5620457973399164), (120, 0.6721578198088268), (150, 0.6984651711924437), (200, 0.6809496236247824), (300, 0.518887505188875), (1000, 0.3119956966110817), (1500, 0.1818181818181818), (2000, 0.13636363636363635), (4000, 0.03305785123966942)],
    # 'KCAE': [(100, 0.517573929338634), (120, 0.6815131177547284), (150, 0.7079102715466347), (200, 0.7348002316155173), (300, 0.6832710668327107), (1000, 0.6503496503496502), (1500, 0.6969696969696969), (2000, 0.8522727272727273), (4000, 0.42975206611570244)],

    # }


    # plot_info_retrieval_by_length(precisions, sys.argv[1])
    plot_info_retrieval(precisions, sys.argv[1])

    # # Effect of number of topics
    # x = [20, 32, 64, 128, 256, 512, 1024, 1500]
    # y = [0.546, 0.694, 0.719, 0.744, 0.747, 0.761, 0.767, 0.713]
    # plot(x, y, 'Number of topics', 'Classification accuracy', sys.argv[1])

    # Effect of alpha
    # x = [0.0625, 0.3, 1, 3, 6, 9, 12]
    # y = [0.711, 0.706, 0.739, 0.738, 0.743, 0.746, 0.743]
    # plot(x, y, r'$\alpha$', 'Classification accuracy', sys.argv[1])

    # # # Effect of k
    # x = [2, 4, 6, 8, 16, 32, 64, 96, 128]
    # y = [0.729, 0.728, 0.720, 0.738, 0.737,  0.744, 0.739, 0.733, 0.714]
    # plot(x, y, r'$k$', 'Classification accuracy', sys.argv[1])

    # # scalability
    # x = [0.5, 1, 1.5, 2]
    # y = [6 , 12.3, 15.8,  49.5]
    # plot(x, y, r'training set size', 'runtime (h)', sys.argv[1])
