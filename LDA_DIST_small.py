import pickle as pkl
from tqdm import tqdm, trange
from random import random, randint
from time import sleep

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
    return clf
def attack(data, clf):
    labels = data['labels']
    traces = data['traces']
    score = clf.score(traces, labels)
    # print(f"SCORE: {score}")
    return score

def HW(x):
    fbin = np.vectorize(np.binary_repr)
    bin_x = [fbin(xi) for xi in x]
    fcount = np.vectorize(np.char.count)
    hw = np.array([fcount(xi, '1') for xi in bin_x])
    return hw
def gen_data_m(states, n_samples, q, n_random=3, noise=0.1):
    data = {}
    dim = states.shape[1]
    labels = np.concatenate((np.ones((n_samples + n_samples%2)//2, dtype=np.int16), np.zeros((n_samples-(n_samples + n_samples%2)//2), dtype=np.int16)), axis=0)
    np.random.shuffle(labels)
    sts = states[labels]
    x0 = np.random.randint(0, q, size=(n_samples, dim))
    x1 = (sts - x0)%q
    L0 = HW(x0) + np.random.normal(0, noise, (n_samples, dim))
    L1 = HW(x1) + np.random.normal(0, noise, (n_samples, dim))
    # L = np.concatenate((L0, L1), axis=1)
    L = L0*L1
    # L = np.hstack([L, np.random.randn(n_samples, n_random)])
    data['labels'] = labels
    data['traces'] = L
    return data


if __name__ == '__main__':
    # states_ = []
    # for i in range(-2, 3):
    #     for j in range(-2, 3):
    #         for t in range(-2, 3):
    #             for s in range(-2, 3):
    #                 states = np.array([i, j, t, s]).reshape((2, 2))
    #                 states_.append(states)
    # states_ = np.array(states_)
    # with open("states_2.npy", "wb") as f:
    #     np.save(f, states_)
    with open("states_2.npy", "rb") as f:
        states_set = np.load(f)
    # print(states_set.shape)
    q = 19
    n_profilings = [1000]
    n_attack = 1000
    for n_profiling in n_profilings:
        sum = {}
        sum['states'] =  []
        sum['scores'] = []
        with trange(states_set.shape[0]) as t:
            for i in t:
                states_pair = states_set[i]
                t.set_description(f'Process {n_profiling}')
                sum['states'].append(states_pair)
                p_data = gen_data_m(states_pair, n_samples=n_profiling, q=q)
                # print(p_data['traces'].shape)
                clf = profiling(p_data)
                a_data = gen_data_m(states_pair, n_samples=n_attack, q=q)
                score = attack(a_data, clf)
                sum['scores'].append(score)
                # t.update()
        # with open(f"sum_{q}_{n_profiling}.pkl", "wb") as f:
        #     pkl.dump(sum, f)

    # for i in tqdm(range(0, batch_num), desc ="Generating"):
    # for states_pair, i in enumerate(states_set):
    #     sum['states'][i] = states_pair
    #     p_data = gen_data_m(states_pair, n_samples=n_profiling, q=q)
    #     clf = profiling(p_data)
    #     a_data = gen_data_m(states_pair, n_samples=n_attack, q=q)
    #     sum['scores'][i] = (attack(a_data, clf))
    # with open("sum_2.pkl", "wb") as f:
    #     pkl.dump(f, sum)
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
    # pbar = tqdm(["a", "b", "c", "d"])
    # for char in pbar:
    #     sleep(0.25)
    #     pbar.set_description("Processing %s" % char)
    #     print(char)
    # with trange(10) as t:
    #     for i in t:
    #         # Description will be displayed on the left
    #         t.set_description('GEN %i' % i)
    #         # Postfix will be displayed on the right,
    #         # formatted automatically based on argument's datatype
    #         t.set_postfix(loss=random(), gen=randint(1,999), str='h',
    #                       lst=[1, 2])
    #         sleep(0.1)
    # with tqdm(total=10, bar_format="{postfix[0]} {postfix[1][value]:>8.2g}",
    #       postfix=["Batch", dict(value=0)]) as t:
    #     for i in range(10):
    #         sleep(0.1)
    #         t.postfix[1]["value"] = i / 2
    #         t.update()
