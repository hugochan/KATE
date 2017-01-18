'''
Created on Dec, 2016

@author: hugo

'''
from __future__ import absolute_import

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy import interpolate


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
    ticks = zip(*precisions.values()[0])[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions.iteritems():
        fr, pr = zip(*val)
        plt.plot(new_x, pr, linestyle='-', alpha=0.6, marker=markers[i],
                        markersize=6, label=model_name)
        i += 1
        # plt.legend(model_name)
    plt.xlabel('Fraction of Retrieved Documents (%)')
    plt.ylabel('Precision (%)')
    legend = plt.legend(loc='upper right', shadow=True)
    plt.savefig(save_file)
    plt.show()

def plot_info_retrieval_by_length(precisions, save_file):
    markers = ["o", "v", "8", "s", "p", "*", "h", "H", "+", "x", "D"]
    ticks = zip(*precisions.values()[0])[0]
    plt.xticks(range(len(ticks)), ticks)
    new_x = interpolate.interp1d(ticks, range(len(ticks)))(ticks)

    i = 0
    for model_name, val in precisions.iteritems():
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

if __name__ == '__main__':
    import sys
    # 20news_retrieval_128D
    # precisions = {
    #     'LDA': [(0.001, 0.4426326461642425), (0.002, 0.41994158258098285), (0.005, 0.38933170851984034), (0.01, 0.36552361840219), (0.02, 0.33586314908895326), (0.05, 0.2728201561244453), (0.1, 0.2070775654290608), (0.2, 0.14147130803649285), (0.5, 0.07701982354516679), (1.0, 0.05050301672031405)],
    #     'DBN': [(0.001, 0.535038381692656), (0.002, 0.5077608265340592), (0.005, 0.465912108337758), (0.01, 0.4264154357337848), (0.02, 0.37657322856108594), (0.05, 0.29198182151435376), (0.1, 0.2197600288870639), (0.2, 0.15325609847145583), (0.5, 0.08605947016611057), (1.0, 0.05050301672031403)],
    #     'DocNADE': [(0.001, 0.5718148022980761), (0.002, 0.5435414956790445), (0.005, 0.5074230900538642), (0.01, 0.4746133312027964), (0.02, 0.43102761550716634), (0.05, 0.3383512940656766), (0.1, 0.25088957318799715), (0.2, 0.16893256617330504), (0.5, 0.0898931631614369), (1.0, 0.05050301672031405)],
    #     'NVDM': [(0.001, 0.05129628735576586), (0.002, 0.0513143919277744), (0.005, 0.0513784045216623), (0.01, 0.05057947447821584), (0.02, 0.04999729766565648), (0.05, 0.05015274063700316), (0.1, 0.050297158296132634), (0.2, 0.05053768818029844), (0.5, 0.050492220758456205), (1.0, 0.05050301672031404)],
    #     'AvgWord2Vec': [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)],
    #     'AE': [(0.001, 0.22827451359049142), (0.002, 0.18935571863080838), (0.005, 0.14794495865260598), (0.01, 0.12336861250406352), (0.02, 0.10404868431566056), (0.05, 0.08489066120247557), (0.1, 0.07413015988839661), (0.2, 0.06568426232571814), (0.5, 0.0563362391994616), (1.0, 0.05050301672031405)],
    #     'VAE': [(0.001, 0.1625428474870794), (0.002, 0.129049389272433), (0.005, 0.09866000303467093), (0.01, 0.0835667523580887), (0.02, 0.07314337881088674), (0.05, 0.06373537802133396), (0.1, 0.05873448646810881), (0.2, 0.05491787941153268), (0.5, 0.05159197722972036), (1.0, 0.05050301672031405)],
    #     'KSAE': [(0.001, 0.23964418481146235), (0.002, 0.20264447448462075), (0.005, 0.16342178135194532), (0.01, 0.1395696943777457), (0.02, 0.12070563824438621), (0.05, 0.09967618984957147), (0.1, 0.08657995851945419), (0.2, 0.07516400405132624), (0.5, 0.061121338068411066), (1.0, 0.05050301672031405)],
    #     'KCAE': [(0.001, 0.5543982040264583), (0.002, 0.5213392555399969), (0.005, 0.4739445034519384), (0.01, 0.4347574243698827), (0.02, 0.3869114198299623), (0.05, 0.30403564261511706), (0.1, 0.2277761656366975), (0.2, 0.15699569840064684), (0.5, 0.08683891514289452), (1.0, 0.05050301672031405)]}

    # 20news_retrieval_512D
    # precisions = {
    # 'LDA': [(0.001, 0.39885579104909563), (0.002, 0.3731774730845299), (0.005, 0.3398807943251651), (0.01, 0.3116261473171748), (0.02, 0.2764053313531843), (0.05, 0.215810770799768), (0.1, 0.1643580962899014), (0.2, 0.11792391367125364), (0.5, 0.0712356757535181), (1.0, 0.05050301672031405)],
    # 'DBN': [(0.001, 0.5553034326268522), (0.002, 0.5285147009124683), (0.005, 0.488347337076094), (0.01, 0.45024297510562405), (0.02, 0.4033891972422051), (0.05, 0.31771321417997606), (0.1, 0.23800250085341837), (0.2, 0.1643866217959289), (0.5, 0.08997211449990615), (1.0, 0.050503016720313966)],
    # 'DocNADE': [(0.001, 0.5771737556124188), (0.002, 0.5443682711340714), (0.005, 0.5036226386465347), (0.01, 0.47000408874935945), (0.02, 0.42806973432528334), (0.05, 0.3391154672218619), (0.1, 0.2523257091581672), (0.2, 0.16856842576301645), (0.5, 0.08886419064880063), (1.0, 0.05050301672031405)],
    # 'NVDM':  [(0.001, 0.051827354801331486), (0.002, 0.05166441365326164), (0.005, 0.05022143615810757), (0.01, 0.050504279087694), (0.02, 0.04984220717270456), (0.05, 0.050210312107871934), (0.1, 0.05031805352276878), (0.2, 0.050525479733275175), (0.5, 0.05048740951458409), (1.0, 0.05050301672031404)],
    # 'AvgWord2Vec': [(0.001, 0.4619200502100128), (0.002, 0.4201226283010617), (0.005, 0.363601016614824), (0.01, 0.3199258385461033), (0.02, 0.27425227583548745), (0.05, 0.2083981501934101), (0.1, 0.15890115524777892), (0.2, 0.1170970848576276), (0.5, 0.0746015280886044), (1.0, 0.05050301672031405)],
    # 'AE':  [(0.001, 0.25062762516293363), (0.002, 0.2055713802925667), (0.005, 0.15574975343297148), (0.01, 0.12843020222861343), (0.02, 0.10797588107849927), (0.05, 0.08867698410088286), (0.1, 0.07749710871105607), (0.2, 0.06824739056183722), (0.5, 0.057956525318736705), (1.0, 0.05050301672031405)],
    # 'VAE':  [(0.001, 0.2864384685945949), (0.002, 0.22214913339448517), (0.005, 0.15730739321751025), (0.01, 0.12415463932062157), (0.02, 0.10121123325141194), (0.05, 0.08010235972535741), (0.1, 0.06876865603310822), (0.2, 0.06069940080002707), (0.5, 0.05329987492643488), (1.0, 0.05050301672031405)],
    # 'KSAE': [(0.001, 0.2766257905663044), (0.002, 0.23146091826389079), (0.005, 0.18327753964039104), (0.01, 0.15379102261032626), (0.02, 0.13065375342492558), (0.05, 0.10653729926356474), (0.1, 0.09147214149778002), (0.2, 0.07821235936221227), (0.5, 0.0621660351341905), (1.0, 0.05050301672031405)],
    # 'KCAE': [(0.001, 0.5370057451841842), (0.002, 0.49623424902234925), (0.005, 0.4398091950534882), (0.01, 0.39517410082761617), (0.02, 0.3440230238886337), (0.05, 0.26452986431933806), (0.1, 0.19919513465212774), (0.2, 0.14045712651660616), (0.5, 0.08193839335997657), (1.0, 0.05050301672031405)]}

    precisions = {
    'DocNADE': [(100, 0.5620457973399164), (120, 0.6721578198088268), (150, 0.6984651711924437), (200, 0.6809496236247824), (300, 0.518887505188875), (1000, 0.3119956966110817), (1500, 0.1818181818181818), (2000, 0.13636363636363635), (4000, 0.03305785123966942)],
    'KCAE': [(100, 0.517573929338634), (120, 0.6815131177547284), (150, 0.7079102715466347), (200, 0.7348002316155173), (300, 0.6832710668327107), (1000, 0.6503496503496502), (1500, 0.6969696969696969), (2000, 0.8522727272727273), (4000, 0.42975206611570244)],

    }


    plot_info_retrieval_by_length(precisions, sys.argv[1])
    # plot_info_retrieval(precisions, sys.argv[1])
