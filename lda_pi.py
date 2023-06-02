import numpy as np
# import matplotlib as mpl
from matplotlib import pyplot as plt
# from scipy.interpolate import make_interp_spline
from helpers import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from tqdm import tqdm, trange
from time import time
from multiprocessing import Pool, Manager, Value, Array, Lock
from functools import partial

def classify(states, n_shares, sigma, q, combif, n_profiling, n_attack):
    labels = np.random.randint(0, 2, n_profiling)
    states_ = states[labels]
    shares = gen_shares(states_, n_shares=n_shares, q=q)

    if combif is None:
        L_shares = gen_leakages(shares, sigma)
        n_coeffs = states_.shape[1]
        L = np.empty((n_profiling, n_coeffs*n_shares))
        for i in range(n_shares):
            L[:, n_coeffs*i:(n_coeffs*i+n_coeffs)] = L_shares[i]
    else:
        L = gen_leakages(shares, sigma, combif)
    cls = LDA()
    cls.fit(L, labels)
    chosen_state = np.random.randint(0, 2, 1)
    chosen_state_arr = np.repeat(states[chosen_state], n_attack, axis=0)
    shares_a = gen_shares(chosen_state_arr, n_shares=n_shares, q=q)
    # L_a = gen_leakages(shares_a, sigma=sigma, combif=combif)
    if combif is None:
        L_shares = gen_leakages(shares_a, sigma)
        n_coeffs = states_.shape[1]
        L_a = np.empty((n_profiling, n_coeffs*n_shares))
        for i in range(n_shares):
            L_a[:, n_coeffs*i:(n_coeffs*i+n_coeffs)] = L_shares[i]
        else:
            L_a = gen_leakages(shares_a, sigma, combif)
    preds = cls.predict_log_proba(L_a)
    guess = np.argmax(preds.sum(axis=0))
    return 1 if guess == chosen_state[0] else 0

def profiling(states, n_shares, sigma, q, combif, n_profiling):
    labels = np.random.randint(0, 2, n_profiling)
    states_ = states[labels]
    shares = gen_shares(states_, n_shares=n_shares, q=q)
    L = gen_leakages(shares, sigma=sigma, combif=combif)
    if combif is None:
        L = leakage_fix(L)
    cls = LDA()
    cls.fit(L, labels)
    return cls

def attack(cls, states, n_shares, sigma, q, combif, n_attack):
    chosen_state = np.random.randint(0, 2, 1)
    chosen_state_arr = np.repeat(states[chosen_state], n_attack, axis=0)
    shares_a = gen_shares(chosen_state_arr, n_shares=n_shares, q=q)
    L_a = gen_leakages(shares_a, sigma=sigma, combif=combif)
    preds = cls.predict(L_a)
    guess = 1 if np.count_nonzero(preds) > n_attack/2 else 0
    return 1 if guess == chosen_state[0] else 0

def model(cls, states, n_shares, sigma, q, combif, n_attack):
    labels = np.random.randint(0, 2, n_attack)
    states_ = states[labels]
    shares_a = gen_shares(states_, n_shares=n_shares, q=q)
    L_a = gen_leakages(shares_a, sigma=sigma, combif=combif)
    if combif is None:
        L_a = leakage_fix(L_a)
    preds = np.log2(cls.predict_proba(L_a))
    s0 = preds[labels==0]
    p0 = s0[np.arange(s0.shape[0]), 0]
    s1 = preds[labels==1]
    p1 = s1[np.arange(s1.shape[0]), 1]


    ce = np.nansum(p0)/p0.shape[0] +  np.nansum(p1)/p1.shape[0]

    return ce/2, cls.score(L_a, labels)

def compute_pi(combif, n_shares):
    n_coeffs = 256
    q = 3329
    # SIGMA = [0.51672043]
    SIGMA = [1.63401346, 5.16720427]
    sigma_bar = tqdm(enumerate(SIGMA), total=len(SIGMA))

    n_attack = 10000
    for i_s, sigma in sigma_bar:
        plt.figure(figsize=(12, 12))
        sigma_bar.set_description(f"Process for {sigma}")
        if i_s == 0:
            N_p = np.arange(500000, 500000, 40000)
        elif i_s == 1:
            N_p = np.arange(2000000, 4000000, 1000000)
        elif i_s == 2:
            N_p = np.arange(100000, 1000000, 100000)
        res_holder = []
        Np_bar = tqdm(N_p, total=len(N_p))
        n_reps = 20
        for n_profiling in Np_bar:
            Np_bar.set_description(f"Estimate using {n_profiling} traces===========")
            reps_bar = tqdm(np.arange(n_reps), total=n_reps)
            pi = 0
            rep = 0
            for nr in reps_bar:
                reps_bar.set_description(f"{n_profiling} Reps: {nr}")
                states = gen_states(2, n_coeffs)
                cls = profiling(states, n_shares, sigma, q, combif, n_profiling)
                ce, _ = model(cls, states, n_shares, sigma, q, combif, n_attack)
                reps_bar.set_postfix_str(f"{_} {(1 + ce):0.4f}")
                if ce != -np.inf:
                    pi += 1 + ce
                    rep += 1
                    with open(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares_{n_profiling}.txt", "a") as ftxt:
                        np.savetxt(ftxt, [1 + ce])

            res_holder.append(pi/rep if rep != 0 else 0)
        with open(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares.npy", "wb") as f:
            np.save(f, N_p)
            np.save(f, res_holder)
        plt.plot(N_p, res_holder)
        plt.scatter(N_p, res_holder)
        plt.savefig(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares.png")
def compute_pi_max(combif, n_shares):
    n_coeffs = 256
    q = 3329
    # SIGMA = [0.51672043]
    SIGMA = [0.51672043, 1.63401346, 5.16720427]
    sigma_bar = tqdm(enumerate(SIGMA), total=len(SIGMA))
    n_attack = 5000
    for i_s, sigma in sigma_bar:
        plt.figure(figsize=(12, 12))
        sigma_bar.set_description(f"Process for {sigma}")
        if i_s == 0:
            N_p = np.arange(500, 3000, 200)
        elif i_s == 1:
            N_p = np.arange(1000, 10000, 1000)
        elif i_s == 2:
            N_p = np.arange(10000, 200000, 10000)

        N_p = [200000]
        res_holder = []
        Np_bar = tqdm(N_p, total=len(N_p))
        n_reps = 50
        for n_profiling in Np_bar:
            Np_bar.set_description(f"Estimate using {n_profiling} traces")
            reps_bar = tqdm(np.arange(n_reps), total=n_reps)
            pi = 0
            rep = 0
            for nr in reps_bar:
                reps_bar.set_description(f"{n_profiling} Reps: {nr}")
                states = gen_states(2, n_coeffs)
                cls = profiling(states, n_shares, sigma, q, combif, n_profiling)
                ce, _ = model(cls, states, n_shares, sigma, q, combif, n_attack)
                reps_bar.set_postfix_str(f"{_} {(1 + ce):0.4f}")
                if ce != -np.inf:
                    pi += 1 + ce
                    with open(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares_{n_profiling}.txt", "w") as freps:
                        np.savetxt(freps, [1 + ce])
                    rep += 1

        #     res_holder.append(pi/rep if rep != 0 else 0)
        # with open(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares_{N_p}.npy", "wb") as f:
        #     np.save(f, N_p)
        #     np.save(f, res_holder)
        # plt.plot(N_p, res_holder)
        # plt.scatter(N_p, res_holder)
        # plt.savefig(f"sr/pi_{combif}_{sigma:0.4f}_{n_shares}shares_{N_p}.png")

def compute_sr(combif):
    n_coeffs = 256
    n_shares = 2
    q = 3329
    n_profiling = 200000
    # SIGMA = [0.51672043]
    SIGMA = [0.51672043, 1.63401346, 5.16720427]
    snr = [10, 1, 0.1]
    # snr, SIGMA = sigma_snr()
    flog = open(f"sr/sr_log_{n_shares}_{n_profiling}.txt", 'a')
    sigma_bar = tqdm(enumerate(SIGMA), total=len(SIGMA), file=flog)
    for i_s, sigma in sigma_bar:

        plt.figure(figsize=(12, 12))
        sigma_bar.set_description(f"Process for {sigma}")
        if i_s == 0 :
            N_a = np.arange(1, 15, 2)
        elif i_s == 1:
            N_a = np.arange(10, 100, 5)
        else:
            N_a = np.arange(100, 1000, 100)

        res_holder = []
        Na_bar = tqdm(N_a, total=len(N_a), file=flog)
        n_reps = 200
        converge = 0
        for n_attack in Na_bar:
            Na_bar.set_description(f"Attack using {n_attack} traces")
            reps_bar = tqdm(np.arange(n_reps), total=n_reps, file=flog)
            success = 0
            for nr in reps_bar:
                reps_bar.set_description(f"{n_attack} Reps: {nr}")
                states = gen_states(2, n_coeffs)
                success += classify(states, n_shares, sigma, q, combif, n_profiling, n_attack)
                reps_bar.set_postfix_str(f"{sigma:0.4f} {(success/(nr+1)):0.4f}")
            sr = success/n_reps
            res_holder.append(success/n_reps)
            converge += 1 if sr >0.9 else 0
            if converge > 3:
                break
        with open(f"sr/sr_N_{sigma}_{combif}_{n_shares}shares.npy", "wb") as f:
            np.save(f, N_a)
            np.save(f, res_holder)
        plt.plot(N_a, res_holder, label=f"{sigma:0.4f}")
        plt.scatter(N_a, res_holder)
        plt.savefig(f"sr/sr_N_{sigma}_{combif}_{n_shares}shares.png")

def pi (q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif):
    res_holder = []
    n_reps = 50
    reps_bar = tqdm(np.arange(n_reps), total=n_reps)
    for nr in reps_bar:
        reps_bar.set_description(f"{n_profiling} Reps: {nr}")
        states = gen_states(2, n_coeffs)
        cls = profiling(states, n_shares, sigma, q, combif, n_profiling)
        ce, _ = model(cls, states, n_shares, sigma, q, combif, n_attack)
        reps_bar.set_postfix_str(f"{_} {(1 + ce):0.4f}")
        res_holder.append(1+ce)
    return res_holder

if __name__ == '__main__':
    import sys
    try:
        op = sys.argv[1]
    except:
        op = "pi"

    try:
        ft = sys.argv[2]
    except:
        ft = "norm_prod"

    try:
        n_shares = int(sys.argv[3])
    except:
        n_shares = 2

    print(f"=============Process {op} {ft} {n_shares} shares==============")
    if op == "pi_max":
        compute_pi_max(ft, n_shares)
    elif op == "pi":
        compute_pi(ft, n_shares)
    elif op == "sr":
        compute_sr(ft)
    elif op == "pi_sep":
        SIGMA = [0.51672043, 1.63401346, 5.16720427]
        q = 3329
        n_shares = 2
        sigma = SIGMA[0]
        n_coeffs = 256
        n_profiling = 50000
        n_attack = 1000
        combif = "norm_prod"
        res = pi(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif)
        print(np.array(res).mean())
    # compute_pi()
    # sr_sigma()
    # sr_worker(sigma=5, combif="norm_prod")
    # sr_compute("norm_prod")
    # import sys
    # x = sys.argv[1]
    # sr_w_sigma()
