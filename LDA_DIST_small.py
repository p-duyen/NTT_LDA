from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def gen_data(states, n_random, n_samples, noise=0.1):
    """
    L = HW(key)||random
    """
    data = {}
    # labels = np.random.randint(0, states.shape[0], (n_samples, ))
    labels = np.concatenate((np.ones((n_samples + n_samples%2)//2, dtype=np.int16), np.zeros((n_samples-(n_samples + n_samples%2)//2), dtype=np.int16)), axis=0)
    np.random.shuffle(labels)
    X = states[labels]
    X = X + np.random.normal(0, noise, (n_samples, 1))
    X = np.hstack([X, np.random.randn(n_samples, n_random)])
    data['labels'] = labels
    data['traces'] = X
    return data

def profiling(data):
    labels = data['labels']
    traces = data['traces']
    clf = LDA().fit(traces, labels)
    print(f"new shape: {clf.transform(traces).shape}")
    return clf
def attack(data, clf):
    labels = data['labels']
    traces = data['traces']
    print(f"SCORE: {clf.score(traces, labels)}")

def HW(x):
    fbin = np.vectorize(np.binary_repr)
    bin_x = [fbin(xi) for xi in x]
    fcount = np.vectorize(np.char.count)
    hw = np.array([fcount(xi, '1') for xi in bin_x])
    return hw
def gen_data_m(states, n_samples, q, noise=0.1):
    """
    L = HW(r)||HW((r-key)modq)
    """
    data = {}
    dim = states.shape[1]
    labels = np.concatenate((np.ones((n_samples + n_samples%2)//2, dtype=np.int16), np.zeros((n_samples-(n_samples + n_samples%2)//2), dtype=np.int16)), axis=0)
    np.random.shuffle(labels)
    sts = states[labels]
    x0 = np.random.randint(0, q, size=(n_samples, dim))
    x1 = (sts - x0)%q
    L0 = HW(x0) + np.random.normal(0, noise, (n_samples, dim))
    L1 = HW(x1) + np.random.normal(0, noise, (n_samples, dim))

    L = np.concatenate((L0, L1), axis=1)
    data['labels'] = labels
    data['traces'] = L
    return data


if __name__ == '__main__':
    # L = HW(r)||HW((r-key)mod q)

    states = np.array([[1], [3]])
    p_data = gen_data_m(states, n_samples=50000, q=5)
    clf = profiling(p_data)
    a_data = gen_data_m(states, n_samples=100, q=5)
    attack(a_data, clf)

    #L = HW(key)||randoms

    # states = np.array([[1], [5], [6]])
    # states = HW(states)
    # print(states)
    # p_data = gen_data(states, n_random=3, n_samples=50000, noise=0.1)
    # print(p_data['traces'].shape)
    # print(p_data['labels'])
    # print("=================================PROFILING============================")
    # clf = profiling(p_data)
    # a_data = gen_data(states, n_random=3, n_samples=100, noise=0.1)
    # print("=================================ATTACKING============================")
    # attack(a_data, clf)
