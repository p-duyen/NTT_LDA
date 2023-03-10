import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import pickle
# from scalib.modeling import LDAClassifier as LDA
from gen_data import *
from discriminant_analysis_ import LinearDiscriminantAnalysis as lda
import seaborn as sns
from scipy import linalg
import math
from prettytable import PrettyTable
from sympy import *


def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return np.array(mean_vectors)

def within_var(X, y):
    class_labels = np.unique(y)
    prior_prob = 1/len(class_labels)
    n_classes = class_labels.shape[0]
    dim = X.shape[1]
    means_array = comp_mean_vectors(X, y)
    Sw = np.zeros((dim, dim))
    for cls, mean in zip(class_labels, means_array):
        norm_X = X[y == cls] - mean
        Sw_class = np.zeros((dim, dim))
        for row in norm_X:
            row = np.expand_dims(row, axis = 0)
            Sw_class += (row.T)@row
        Sw += Sw_class
    return Sw/X.shape[0]

def between_var(X, y):
    dim = X.shape[1]
    total_mean = np.mean(X, axis=0).reshape(1, dim)
    means_array = comp_mean_vectors(X, y)
    n_classes = np.unique(y)
    Sb = np.zeros((dim, dim))
    for cls in n_classes:
        mean_distance = means_array[cls].reshape(1, dim) - total_mean
        class_samples = X[y==cls].shape[0]
        Sb_class = mean_distance.T@mean_distance
        Sb += class_samples*Sb_class
    return Sb/X.shape[0]

def get_components(eig_vals, eig_vecs, dim, n_comp=2):
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.vstack([eig_pairs[i][1] for i in range(0, n_comp)]).reshape(dim, n_comp)
    return W

def reduce_matrix(Sw, Sb, n_comp=2):
    dim = Sw.shape[1]
    eig_vals, eig_vecs = linalg.eigh(Sb, Sw)
    # print(f"evals: {eig_vals}")
    W = get_components(eig_vals, eig_vecs, dim=dim, n_comp=n_comp)
    return W

def pretty_print(x):
    p = PrettyTable()
    for row in x:
        p.add_row(row)
    print( p.get_string(header=False, border=False))


def reduce_pdf(X, y, n_comp=1):
    Sw = within_var(X, y)
    Sb = between_var(X, y)
    W = reduce_matrix(Sw, Sb, n_comp=n_comp)
    W_T = W.transpose()
    V = np.linalg.inv(Sw)@Sb
    # print("Sw")
    # pretty_print(Sw)
    # print("Sb")
    # pretty_print(Sb)
    # print("invSw")
    # pretty_print(np.linalg.inv(Sw))
    # print("Sw-1Sb")
    # pretty_print(V)

    reduce_X = X.dot(W)

    pooled_cov = within_var(reduce_X, y)

    reduce_means = comp_mean_vectors(reduce_X, y)
    return reduce_means, pooled_cov, W
def pdf_normal(x, means_array, pooled_cov):
    dim = x.shape[1]
    det_cov = np.linalg.det(pooled_cov)
    inv_cov = np.linalg.inv(pooled_cov)
    norm_x = x - means_array
    numerator = norm_x.T@inv_cov
    numerator = -1/2 * numerator@norm_x
    denominator = ((2*np.pi)**(dim/2))*np.sqrt(det_cov)
    return np.exp(numerator)/denominator
def comp_proba_reduced(X, W, reduce_means, pooled_cov, n_classes=2):
    n_samples = X.shape[0]
    reduce_X = X@W
    if n_samples == 1:
        pdfs = np.array([pdf_normal(reduce_X, reduce_means[cls], pooled_cov) for cls in range(n_classes)])
        pdfs = pdfs/np.sum(pdfs)
    else:
        pdfs = []
        for x in reduce_X:
            x = np.expand_dims(x, axis=0)
            pdfx = np.array([pdf_normal(x, reduce_means[cls], pooled_cov) for cls in range(n_classes)])
            pdfx = pdfx/np.sum(pdfx)
            pdfs.append(pdfx)
        pdfs = np.array(pdfs)
    return pdfs.squeeze(), reduce_X

def comp_proba(X, reduce_means, pooled_cov, n_classes=2):
    n_samples = X.shape[0]
    if n_samples == 1:
        pdfs = np.array([pdf_normal(X, reduce_means[cls], pooled_cov) for cls in range(n_classes)])
        pdfs = pdfs/np.sum(pdfs)
    else:
        pdfs = []
        for x in X:
            x = np.expand_dims(x, axis=0)
            pdfx = np.array([pdf_normal(x, reduce_means[cls], pooled_cov) for cls in range(n_classes)])
            pdfx = pdfx/np.sum(pdfx)
            pdfs.append(pdfx)
        pdfs = np.array(pdfs)
    return pdfs.squeeze()
def uni_test():

    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    x0 = np.array([[1., 2.], [2., 3.], [3., 3.], [4., 5.], [5., 5.]])
    x1 = np.array([[4., 2.], [5., 0.], [5., 2.], [3., 2.], [5., 3.], [6., 3.]])
    X = np.vstack((x0, x1))
    mu, cov, W = reduce_pdf(X, labels, n_comp=2)
    # print(f"mu:{mu}, cov:{cov}, W:{W}")


    s = np.random.uniform(-5, 5, size=(500, 1, 1))
    m0 = np.expand_dims(mu[0], axis=0)
    m1 = np.expand_dims(mu[1], axis=0)
    sort_i = np.argmax(s)
    sort_s = np.sort(s, axis=0)
    # print(sort_s)

    # prob_s = comp_proba(s, mu, cov, n_classes=2)
    # f0 = np.array([pdf_normal(s_, m0, cov) for s_ in sort_s])
    # f1 = np.array([pdf_normal(s_, m1, cov) for s_ in sort_s])
    # # f0 = prob_s[:, 0]
    # # f1 = prob_s[:, 1]
    # # reduce_s = np.squeeze(reduce_s)
    # # sns.histplot(data=f0.squeeze(), x=sort_s.squeeze(), kde=True)
    # plt.plot(sort_s.squeeze(), f0.squeeze(), color='green')
    # plt.scatter(sort_s.squeeze(), f0.squeeze(), color='green', marker='o')
    # plt.plot(sort_s.squeeze(), f1.squeeze(), color='purple')
    # l = traces[0].reshape(1, dim)
    # mu, cov, W = reduce_pdf(traces, labels)
    # r_l = l.dot(W)
    # project_x0 = x0.dot(W)
    # project_x1 = x1.dot(W)
    # y0 = np.zeros(5)
    # y1 = np.zeros(6)
    # plt.scatter(project_x0, y0)
    # plt.scatter(project_x1, y1)
    plt.show()
    # pred = comp_proba(r_l, mu, cov, n_classes=2)
    # print(pred.shape, pred)
    # clf = LDA(2, 1, 3)
    # clf.fit_u(traces, labels, 0)
    # print(clf.get_sb())
    # clf.solve()
def proba_test(states_pair, comf):
    batch_size = 100000
    dim = 256
    noise = 0.1
    q = 3329
    labels, traces = gen_am_ho(batch_size, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf=comf)
    # means = comp_mean_vectors(traces, labels)
    # for i in range(dim):
    #     print(f"state {i}: {st0[i]}, {means[0][i]}", end=" ")
    # for i in range(dim):
    #     print(f"state {i}: {st1[i]}, {means[1][i]}", end=" ")


    clf = lda(solver="eigen")
    clf.fit(traces, labels)
    mu, cov, W = reduce_pdf(traces, labels, n_comp=1)
    print(mu[0]-mu[1], cov)
    # print(f"transform : {W}")
    a_labels, a_traces = gen_am_ho(1000, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf=comf)
    reduce_traces = a_traces.dot(W)
    reduce_traces.sort(axis=0)
    reduce_traces = np.expand_dims(reduce_traces, axis=1)
    m0 = np.expand_dims(mu[0], axis=0)
    m1 = np.expand_dims(mu[1], axis=0)
    x0 = a_traces[a_labels==0].dot(W)
    x1 = a_traces[a_labels==1].dot(W)

    y0 = np.ones((x0.shape[0]))*0.0025
    y1 = np.ones((x1.shape[0]))*-0.005
    reduce_traces = np.expand_dims(reduce_traces, axis=1)
    f0 = np.array([pdf_normal(reduce_trace, m0, cov) for reduce_trace in reduce_traces])
    f1 = np.array([pdf_normal(reduce_trace, m1, cov) for reduce_trace in reduce_traces])

    fig, ax = plt.subplots()
    print(clf.score(a_traces, a_labels))
    lines = plt.plot(reduce_traces.squeeze(), f0.squeeze(), color='green', label="class0")
    ax.plot(reduce_traces.squeeze(), f1.squeeze(), color='purple', label="class1")
    ax.scatter(x0, y0, color="green", s=5, alpha=0.5)
    ax.scatter(x1, y1, color="purple", s=5, alpha=0.5)
    ax.scatter(m0, [0.0025], color="green", marker="x",s=100, label="mean cls0")
    ax.scatter(m1, [-0.005], color="purple", marker="x",s=100, label="mean cls1")
    axs[i].text(np.round(m0, 2)[0][0], 0.004, f"{np.round(m0, 3)[0][0]}", color="green")
    axs[i].text(np.round(m1, 2)[0][0], -0.004, f"{np.round(m1, 3)[0][0]}", color="purple")
    # #
    # #
    # print(np.round(m0, 2)[0][0])
    # [xmin, xmax] = ax.get_xlim()
    # print( ax.get_xlim()[0],  ax.get_xlim()[1])
    # ticks = [round( ax.get_xlim()[0], 3), np.round(m0, 3)[0][0], 0, np.round(m1, 3)[0][0],round( ax.get_xlim()[1], 3)]
    # ax.set_xticks(ticks)
    # ax.get_xticklabels()[0].set_color("green")
    # ax.get_xticklabels()[1].set_color("purple")
    plt.legend()
    plt.title(f"{comf}_score: {clf.score(a_traces, a_labels)}")
    plt.show()


def f_compare(comf, states_pair):
    batch_size = 100000
    dim = 256
    noise = 0.1
    q = 3329
    # st0 = np.random.randint(-2, 3, (dim, 1))
    # st1 = np.random.randint(-2, 3, (dim, 1))
    # states_pair = np.hstack((st0, st1)).T
    # mng = plt.get_current_fig_manager()
    # mng.resize(*mng.window.maxsize())
    fig, axs = plt.subplots(1, 3)
    fig.suptitle(f"noise = {noise}", fontsize=12, y=0.55)
    for i, f in enumerate(comf):
        print(f"=========================={f}===========================")
        labels, traces = gen_am_ho(batch_size, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf=f)
        clf = lda(solver="eigen")
        clf.fit(traces, labels)
        mu, cov, W = reduce_pdf(traces, labels, n_comp=1)
        a_labels, a_traces = gen_am_ho(1000, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf=f)
        reduce_traces = a_traces.dot(W)
        reduce_traces.sort(axis=0)
        reduce_traces = np.expand_dims(reduce_traces, axis=1)
        m0 = np.expand_dims(mu[0], axis=0)
        m1 = np.expand_dims(mu[1], axis=0)
        x0 = a_traces[a_labels==0].dot(W)
        x1 = a_traces[a_labels==1].dot(W)

        y0 = np.ones((x0.shape[0]))*0.0025
        y1 = np.ones((x1.shape[0]))*-0.005
        reduce_traces = np.expand_dims(reduce_traces, axis=1)
        f0 = np.array([pdf_normal(reduce_trace, m0, cov) for reduce_trace in reduce_traces])
        f1 = np.array([pdf_normal(reduce_trace, m1, cov) for reduce_trace in reduce_traces])

        #
        score = clf.score(a_traces, a_labels)
        axs[i].plot(reduce_traces.squeeze(), f0.squeeze(), color='green', label="class0")
        axs[i].plot(reduce_traces.squeeze(), f1.squeeze(), color='purple', label="class1")
        axs[i].scatter(x0, y0, color="green", s=5, alpha=0.5)
        axs[i].scatter(x1, y1, color="purple", s=5, alpha=0.5)
        axs[i].scatter(m0, [0.0025], color="green", marker="x",s=100, label="mean cls0")
        axs[i].scatter(m1, [-0.005], color="purple", marker="x",s=100, label="mean cls1")
        axs[i].text(np.round(m0, 2)[0][0], 0.004, f"{np.round(m0, 3)[0][0]}", color="green")
        axs[i].text(np.round(m1, 2)[0][0], -0.004, f"{np.round(m1, 3)[0][0]}", color="purple")
        # [xmin, xmax] = ax.get_xlim()
        # ticks = [np.round(m0, 2)[0][0], np.round(m1, 3)[0][0]]
        # axs[i].set_xticks(ticks)
        # axs[i].get_xticklabels()[0].set_color("green")
        # axs[i].get_xticklabels()[1].set_color("purple")
        axs[i].set_title(f"{f} score: {score}")
        # for i in range(dim):
        #     print(f"{W[i]} {st0[i]} {st1[i]}")
        pretty_print(mu)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    # uni_test()
    dim = 256
    st0 = np.random.randint(-1, 1, (dim, 1))
    st1 = np.random.randint(-1, 1, (dim, 1))
    states_pair = np.hstack((st0, st1)).T
    #
    # comf = ["prod", "norm_prod", "abs_diff"]
    # comf = ["norm_prod", "abs_diff", "prod"]
    # [proba_test(states_pair, f) for f in comf]
    comf = ["abs_diff", "norm_prod", "prod"]
    f_compare(comf, states_pair)
    # l = traces[0].reshape(1, dim)
