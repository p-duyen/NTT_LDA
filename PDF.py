import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
import seaborn as sns
import matplotlib.ticker as ticker
from stat__ import am_gen, HW

def pdf_normal(x, mu, var):
    return np.exp(-(x-mu)**2/(2*var**2)) / (var * np.sqrt(2*np.pi))


# def marginal_f(l, range, var):
#     f_ = 0
#     for i in range(range):
#         f_ += norm.pdf(l, HW(i), var)
#     f_ = f_/256
#     return f_

def pdf_l_given_s(s, l, x_range, var):
    """
    P(L=[l0, l1]|S) for 1st order masking
    (L_i|X_i) ~ normal(HW(X_i), var)
    """
    f_ = 0
    for x in range(x_range):
        ms = (s-x)%x_range
        f_ += pdf_normal(l[:, 0], HW(x), var)*pdf_normal(l[:, 1], HW(ms), var)
    f_ = f_/x_range
    return f_


pdf_l_given_s_in_S = np.vectorize(pdf_l_given_s, excluded=[1])

def pdf_s_given_l(s, l, s_range, x_range, var):
    """P(S=s| L = [l0, l1]) for 1st order masking
    (L|X) ~ normal(HW(W), var)
    """
    sum_pdf = np.sum(pdf_l_given_s_in_S(s_range, l, x_range, var))
    f_ = pdf_l_given_s(s, l, x_range, var)/sum_pdf
    return f_

def pdf_S_given_l(l, s_range, x_range, var):
    """P(S| L = [l0, l1]) for 1st order masking for each value s in S
    (L_i|X_i) ~ normal(HW(X_i), var)
    """
    # S_range = np.arange(s_range)
    pdf_S_given_l = pdf_l_given_s_in_S(s_range, l, x_range, var)
    f_ = pdf_S_given_l/np.sum(pdf_S_given_l)
    return f_

def uni_test():
    n_samples = 500
    noise = 0.2
    s_range = [-2, -1, 0, 1, 2]
    q = 3329
    for s in s_range:
        print(f"================{s}===============")
        st = np.array([s])
        X, L = am_gen(500, st, q=q, noise=noise, comf=None)
        for l in L:
            print(pdf_S_given_l(l, s_range, q, noise))

def partial_test():
    q = 5
    noise = 0.1
    s = np.array([0])
    s_range = [-2, -1, 0, 1, 2]
    r = np.random.randint(0, q, (1))
    # r = np.array([0])
    ms = (s-r)%q
    l0 = HW(r) + np.random.normal(0, noise, (1, ))
    l1 = HW(ms) + np.random.normal(0, noise, (1, ))
    l = np.column_stack((l0, l1))
    y = np.column_stack((r, ms))
    print(f"fixed random: {r}")
    print(f"fix secret: {s}")
    print(pdf_l_given_s_in_S(s_range, l, q, noise))
    for s_i in s_range:
        print(f"running secret: {s_i}")
        # print(f"l given s={s_i}", pdf_l_given_s(s_i, l, q, noise))
        print(f"s={s_i} given l", pdf_s_given_l(s_i, l, s_range, q, noise))
    print(pdf_S_given_l(l, s_range, q, noise))

def multi_test():
    q = 3329
    noise = 0.1
    dim = 256
    # s = np.array([0, 2, 1])
    s = np.random.randint(-2, 3, (dim, ))
    s_range = [-2, -1, 0, 1, 2]
    r = np.random.randint(0, q, (dim, ))
    ms = (s-r)%q
    l0 = HW(r) + np.random.normal(0, noise, (dim, ))
    l1 = HW(ms) + np.random.normal(0, noise, (dim, ))
    l = np.column_stack((l0, l1))
    y = np.column_stack((r, ms))
    # print(l, y)
    print(f"fix secret: {s}")
    print(f"fixed random: {r}")
    s_ = np.random.randint(-2, 3, (dim, ))
    proba_matrix = multi_dim_proba(s, l, dim, s_range, q, noise)
    proba_matrix_ = multi_dim_proba(s_, l, dim, s_range, q, noise)
    # s_idx = [np.where(np.array(s_range)==s_i)[0][0] for s_i in s]
    print(proba_matrix)
    print(proba_matrix_)

def multi_trace_acc_proba(s, L, dim, s_range, x_range, var):
    acc_proba = [multi_dim_proba(s, l, dim, s_range, q, noise) for l in L]
    acc_proba = np.array(acc_proba)
    return np.sum(acc_proba)


def distin_pairs():
    q = 3329
    noise = 1
    dim = 256
    s_range = [-2, -1, 0, 1, 2]
    sr = 0
    n_query = 5
    for i in range(n_query):
        print('================',i,'====================')
        coin_flip = np.random.randint(0, 1, (1))
        s0 = np.random.randint(-2, 3, (dim, 1))
        s1 = np.random.randint(-2, 3, (dim, 1))
        states_pair = np.hstack((s0, s1)).T
        # print(states_pair, states_pair.shape)
        chosen_state = states_pair[coin_flip]
        # print(chosen_state)
        r = np.random.randint(0, q, (dim, 1))
        ms = (chosen_state-r)%q
        l0 = HW(r) + np.random.normal(0, noise, (dim, 1))
        l1 = HW(ms) + np.random.normal(0, noise, (dim, 1))
        l = np.column_stack((l0, l1))
        y = np.column_stack((r, ms))
        proba_pair = [multi_dim_proba(s, l, dim, s_range, q, noise) for s in states_pair]
        proba_pair = np.array(proba_pair)
        pred_state = np.argmax(proba_pair)
        if chosen_state == pred_state:
            sr += 1
        print(f"chosen_state: {chosen_state}, pred_state: {pred_state}")
    sr = sr/n_query





def multi_dim_proba(S, L, dim, s_range, x_range, var):
    proba_total = []
    s_idx = [np.where(np.array(s_range)==s_i)[0][0] for s_i in S]
    for l, s in zip(L, S):
        l = np.expand_dims(l, axis=0)
        # p = pdf_S_given_l(l, s_range, x_range, var)
        # print(p)
        proba_total.append(pdf_S_given_l(l, s_range, x_range, var))
    proba_total = np.array(proba_total)
    proba_s = proba_total[np.arange(dim), s_idx]
    return np.sum(np.log(proba_s))





if __name__ == '__main__':
    # uni_test()
    # partial_test()
    # multi_test()
    distin_pairs()
    # x_range = np.arange(3329)
    # hw_x = HW(x_range)
    # unique_hw = np.unique(hw_x, return_counts=True)
    # print(unique_hw)


    # x = np.array([0, 1, 2])

    # print(pdf_normal(x, 0, 1))
