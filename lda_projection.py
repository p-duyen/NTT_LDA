import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
import pickle
# from scalib.modeling import LDAClassifier as LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from helpers import *
from discriminant_analysis_ import LinearDiscriminantAnalysis as lda
from scipy import linalg
import math
from prettytable import PrettyTable
from sympy import *

def mean_vec(data, labels):
    """Compute means vector for each class
    """
    class_labels = np.unique(labels)
    mu = np.empty((class_labels.shape[0], data.shape[1]))

    for i in class_labels:
        mu[i] = data[labels==i].mean(axis=0, keepdims=True)
    return mu

def scatter_within(data, labels):
    """Compute within classes scatter matrix:
    S_w = sum_{class_i}S_i
    S_i = cov matrix (scaled by n)
    """
    class_labels = np.unique(labels)
    n_coeff = data.shape[1]
    mu_classes = mean_vec(data, labels)
    S_w = np.zeros((n_coeff, n_coeff))
    for i in class_labels:
        norm_Xi = data[labels==i] - mu_classes[i]
        for row in norm_Xi:
            row = np.expand_dims(row, 1)
            S_w += row@row.T
    return S_w/data.shape[0]
def scatter_between(data, labels):
    class_labels = np.unique(labels)
    n_coeff = data.shape[1]
    mu_classes = mean_vec(data, labels)
    mu_total = data.mean(axis=0, keepdims=True)
    S_b = np.zeros((n_coeff, n_coeff))
    for i in class_labels:
        mean_diff = mu_classes[i]-mu_total
        S_b += (mean_diff.T@mean_diff)*(data[labels==i].shape[0])
    return S_b/data.shape[0]

def project_vector(Sw, Sb, n_components=1):
    eig_vals, eig_vecs = linalg.eigh(Sb, Sw)
    eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    V = np.vstack([eig_pairs[i][1] for i in range(0, n_components)]).reshape(Sw.shape[0], n_components)
    return V
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

def run():
    q = 3329
    n_profiling = 100000
    n_coeff = 256
    n_shares = 2
    sigma = 0.51672043
    combif = None
    states_pair = gen_states(2, n_coeff)

    labels = np.random.randint(0, 2, n_profiling)
    # labels = np.zeros((n_profiling,), dtype=np.int8)
    # ones = np.random.randint(0, n_profiling, n_profiling//2)
    # labels[ones] = 1
    states = states_pair[labels]
    shares = gen_shares(states, n_shares, q)
    data = gen_leakages(shares, sigma=sigma, combif=combif)
    data = leakage_fix(data)

    Sb = scatter_between(data, labels)
    Sw = scatter_within(data, labels)
    W = project_vector(Sw, Sb, 2)
    # Scatter group
    Y = data.dot(W)
    # print(Y.shape)
    Y0 = Y[labels==0]
    Y1 = Y[labels==1]

    plt.scatter(Y0[:, 0], Y0[:, 1], label="class0", s=10)
    plt.scatter(Y1[:, 0], Y1[:, 1], label="class1", s=10, alpha=0.5)


    # V = project_vector(Sw, Sb)
    #
    # Y = data.dot(V)
    # Y0 = Y[labels==0]
    # Y1 = Y[labels==1]
    # # LDA projection
    # p0, x0 = np.histogram(Y0, bins=500, density=True)
    # plt.plot(x0[:-1], p0, label="class0 projected", alpha=0.75)
    # p1, x1 = np.histogram(Y1, bins=500, density=True)
    # plt.plot(x1[:-1], p1, label="class1 projected", alpha=0.75)
    #
    # # LDA classification
    # pooled_cov = scatter_within(Y, labels)
    # print(pooled_cov)
    # mu_projected = mean_vec(Y, labels)
    # gp_0 = pdf_normal(np.sort(Y0, axis=0), mu_projected[0], pooled_cov)
    # gp_1 = pdf_normal(np.sort(Y1, axis=0), mu_projected[1], pooled_cov)
    # plt.plot(np.sort(Y0, axis=0), gp_0, label="class 0 Gaussian estimation", color="tab:blue")
    # plt.plot(np.sort(Y1, axis=0), gp_1, label="class 1 Gaussian estimation", color="tab:orange")

    plt.title(f"{combif}")
    plt.legend()
    plt.show()



run()
