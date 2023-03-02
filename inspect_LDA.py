import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import pickle
from scalib.modeling import LDAClassifier as LDA
from gen_data import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as lda


def comp_mean_vectors(X, y):
    class_labels = np.unique(y)
    n_classes = class_labels.shape[0]
    mean_vectors = []
    for cl in class_labels:
        mean_vectors.append(np.mean(X[y==cl], axis=0))
    return np.array(mean_vectors)

# def scatter_within(X, y):
#     class_labels = np.unique(y)
#     n_classes = class_labels.shape[0]
#     n_features = X.shape[1]
#     mean_vectors = comp_mean_vectors(X, y)
#     S_W = np.zeros((n_features, n_features))
#     for cl, mv in zip(class_labels, mean_vectors):
#         class_sc_mat = np.zeros((n_features, n_features))
#         for row in X[y == cl]:
#             row, mv = row.reshape(n_features, 1), mv.reshape(n_features, 1)
#             class_sc_mat += (row-mv).dot((row-mv).T)
#         S_W += class_sc_mat
#     return S_W

def within_var(X, y):
    class_labels = np.unique(y)
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
    return Sw

# def scatter_between(X, y):
#     overall_mean = np.mean(X, axis=0)
#     n_features = X.shape[1]
#     mean_vectors = comp_mean_vectors(X, y)
#     S_B = np.zeros((n_features, n_features))
#     for i, mean_vec in enumerate(mean_vectors):
#         n = X[y==i+1,:].shape[0]
#         mean_vec = mean_vec.reshape(n_features, 1)
#         overall_mean = overall_mean.reshape(n_features, 1)
#         S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
#     return S_B

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
    return Sb

def get_components(eig_vals, eig_vecs, dim, n_comp=2):
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    W = np.vstack([eig_pairs[i][1] for i in range(0, n_comp)]).reshape(dim, n_comp)
    return W

def reduce_matrix(Sw, Sb, n_comp=2):
    dim = Sw.shape[1]
    eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    W = get_components(eig_vals, eig_vecs, dim=dim, n_comp=n_comp)
    return W

def reduce_pdf(X, y, n_comp=1):
    Sw = within_var(X, y)
    Sb = between_var(X, y)
    W = reduce_matrix(Sw, Sb, n_comp=n_comp)
    W_T = W.transpose()

    pooled_cov = W_T@Sw
    pooled_cov = pooled_cov@W

    reduce_X = X.dot(W)
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
    batch_size = 100000
    dim = 10
    noise = 0.1
    q = 5
    st0 = np.random.randint(-2, 3, (dim, 1))
    st1 = np.random.randint(-2, 3, (dim, 1))
    states_pair = np.hstack((st0, st1)).T
    labels, traces = gen_am_ho(batch_size, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf="norm_prod")
    labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    x0 = np.array([[1., 2.], [2., 3.], [3., 3.], [4., 5.], [5., 5.]])
    x1 = np.array([[4., 2.], [5., 0.], [5., 2.], [3., 2.], [5., 3.], [6., 3.]])
    X = np.vstack((x0, x1))
    s = np.random.uniform(-5, 5, size=(1000, 1, 1))
    mu, cov, W = reduce_pdf(X, labels, n_comp=1)
    m0 = np.expand_dims(mu[0], axis=0)
    m1 = np.expand_dims(mu[1], axis=0)

    prob_s = comp_proba(s, mu, cov, n_classes=2)
    # f0 = [pdf_normal(s_, m0, cov) for s_ in s]
    # f1 = [pdf_normal(s_, m1, cov) for s_ in s]
    f0 = prob_s[:, 0]
    f1 = prob_s[:, 1]
    # reduce_s = np.squeeze(reduce_s)
    plt.scatter(s, f0, color='green')
    plt.scatter(s, f1, color='purple')
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




if __name__ == '__main__':
    # uni_test()
    batch_size = 100000
    dim = 256
    noise = 0.1
    q = 3329
    st0 = np.random.randint(-2, 3, (dim, 1))
    st1 = np.random.randint(-2, 3, (dim, 1))
    states_pair = np.hstack((st0, st1)).T
    labels, traces = gen_am_ho(batch_size, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf="norm_prod")
    mu, cov, W = reduce_pdf(traces, labels, n_comp=1)
    clf = lda()
    clf.fit(traces, labels)
    a_labels, a_traces = gen_am_ho(1000, states_pair, n_order=2, select_state=None, q=q, noise=noise, comf="norm_prod")
    reduce_traces = a_traces.dot(W)
    reduce_traces = np.expand_dims(reduce_traces, axis=1)
    m0 = np.expand_dims(mu[0], axis=0)
    m1 = np.expand_dims(mu[1], axis=0)
    f0 = [pdf_normal(reduce_trace, m0, cov) for reduce_trace in reduce_traces]
    f1 = [pdf_normal(reduce_trace, m1, cov) for reduce_trace in reduce_traces]

    print(clf.score(a_traces, a_labels))
    plt.scatter(reduce_traces, f0, color='green')
    plt.scatter(reduce_traces, f1, color='purple')
    plt.show()
    # l = traces[0].reshape(1, dim)
