from time import time
import gc
from functools import partial
from multiprocessing import Pool, Manager, Value, Array, Lock, Process
from multiprocessing.pool import ThreadPool
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import asyncio
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scalib.modeling import LDAClassifier
from tqdm import tqdm, trange
import numpy as np
from helpers import *

def profiling(secrets, n_shares, sigma, q, combif, n_profiling, n_splits=10):
    splits_pbar = tqdm(range(n_splits), total=n_splits, position=1, leave=False)
    cls = LDAClassifier(2, 1, secrets.shape[1])
    for i_splits in splits_pbar:
        splits_pbar.set_description(f"PROFILING| {i_splits +1}")
        labels = np.random.randint(0, 2, n_profiling//n_splits)
        states = secrets[labels]
        shares = gen_shares(states, n_shares=n_shares, q=q)
        L = gen_leakages(shares, sigma=sigma, combif=combif)*1000
        L = L.astype(np.int16)
        if combif is None:
            L = leakage_fix(L)
        cls.fit_u(L, labels.astype(np.uint16))
        del shares
        del L
        del states
        gc.collect()
    cls.solve()
    return cls

def model(cls, secrets, n_shares, sigma, q, combif, n_attack):
    labels = np.random.randint(0, 2, n_attack)
    states = secrets[labels]
    shares_a = gen_shares(states, n_shares=n_shares, q=q)
    L_a = gen_leakages(shares_a, sigma=sigma, combif=combif)*1000
    L_a = L_a.astype(np.int16)
    if combif is None:
        L_a = leakage_fix(L_a)
    preds = np.log2(cls.predict_proba(L_a))
    s0 = preds[labels==0]
    p0 = s0[np.arange(s0.shape[0]), 0]
    s1 = preds[labels==1]
    p1 = s1[np.arange(s1.shape[0]), 1]
    ce = np.nansum(p0)/p0.shape[0] +  np.nansum(p1)/p1.shape[0]
    return ce/2
def attack(cls, states, n_shares, sigma, q, combif, n_attack):
    chosen_state = np.random.randint(0, 2, 1)
    chosen_state_arr = np.repeat(states[chosen_state], n_attack, axis=0)
    shares_a = gen_shares(chosen_state_arr, n_shares=n_shares, q=q)
    L_a = gen_leakages(shares_a, sigma=sigma, combif=combif)*1000
    L_a = L_a.astype(np.int16)
    preds = cls.predict_proba(L_a)
    preds = np.log10(preds)
    guess = np.argmax(preds.sum(axis=0))
    return 1 if guess == chosen_state[0] else 0


def pi_worker(idx, q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits):
    secrets = gen_states(2, n_coeffs)
    cls = profiling(secrets, n_shares, sigma, q, combif, n_profiling, n_splits)
    ce = model(cls, secrets, n_shares, sigma, q, combif, n_attack)
    return 1 + ce
def compute_pi(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits, n_reps=20, n_proc=20):
    res_holder = []
    rep_chunks = np.array_split(np.arange(n_reps), n_reps//n_proc)
    rep_pbar = tqdm(rep_chunks, total=n_reps, position=0, leave=True)
    for i_rep in rep_pbar:
        rep_pbar.set_description(f"{q} {sigma:0.4f} {n_profiling}")
        f_pi = partial(pi_worker, q=q, sigma=sigma, n_coeffs=n_coeffs, n_shares=n_shares, n_profiling=n_profiling, n_attack=n_attack, combif=combif, n_splits=n_splits)
        results = Parallel(n_jobs=n_proc)(delayed(f_pi)(i) for i in i_rep)
        for val in results:
            res_holder.append(val)
        rep_pbar.update(n_proc-1)
    return res_holder
def compute_pi_seq(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits, n_reps=20, n_proc=5):
    res_holder = []
    rep_pbar = tqdm(np.arange(n_reps), total=n_reps, position=0, leave=True)
    for i_rep in rep_pbar:
        rep_pbar.set_description(f"{q} {sigma:0.4f} {n_profiling}")
        res_holder.append(pi_worker(0, q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits))
    return res_holder

def sr_worker(idx, n_attack, q, sigma, n_coeffs, n_shares, n_profiling, combif, n_splits):
    secrets = gen_states(2, n_coeffs)
    cls = profiling(secrets, n_shares, sigma, q, combif, n_profiling, n_splits)
    guess = attack(cls, secrets, n_shares, sigma, q, combif, n_attack)
    return guess

def compute_sr(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits, n_reps=20, n_proc=20):
    res_holder = []
    rep_chunks = np.array_split(np.arange(n_reps), n_reps//n_proc)
    rep_pbar = tqdm(rep_chunks, total=n_reps, position=0, leave=True)
    for i_rep in rep_pbar:
        rep_pbar.set_description(f"{q} {sigma:0.4f} {n_attack}")
        f_sr = partial(sr_worker, n_attack=n_attack, q=q, sigma=sigma, n_coeffs=n_coeffs, n_shares=n_shares, n_profiling=n_profiling, combif=combif, n_splits=n_splits)
        results = Parallel(n_jobs=n_proc)(delayed(f_sr)(i) for i in i_rep)
        for val in results:
            res_holder.append(val)
        rep_pbar.update(n_proc-1)
    return np.array(res_holder)

def sr_run():
    SIGMA = [0.51672043, 1.63401346, 5.16720427]
    q = 3329
    n_shares = 2
    n_coeffs = 256
    combif = "abs_diff"
    n_profiling = 200000
    Na_1 = np.arange(1, 30, 2)
    Na_2 = np.arange(1, 50, 3)
    Na_3 = np.arange(100, 1000, 50)
    n_splits = 2
    NA = [Na_1, Na_2, Na_3]
    for i_sig, sigma in enumerate(SIGMA):
        n_attacks = NA[i_sig]
        res_holder = []
        fn = f"sr/sr_{sigma:0.4f}_{combif}_{n_profiling}.npy"
        for n_attack in n_attacks:
            n_reps = 100
            res = compute_sr(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits, n_reps=n_reps, n_proc=50)
            res_holder.append(res.mean())
        with open(fn, "wb") as f:
            np.save(f, n_attacks)
            np.save(f, res_holder)
        plt.figure(figsize=(12, 12))
        plt.plot(n_attacks, res_holder)
        plt.scatter(n_attacks, res_holder)
        plt.savefig(f"sr/sr_{sigma:0.4f}_{combif}_{n_profiling}.png")


def pi_run():
    combif = "norm_prod"
    n_attacks = [10000, 50000, 100000]
    Np_1 = np.arange(100000, 900000, 50000)
    Np_2 = np.arange(300000, 1000000, 100000)
    Np_3 = np.arange(50000000, 50000000, 5000000)
    NP = [Np_1, Np_2, Np_3]
    NSPLITS = [10, 20, 200]
    for i_sig, sigma in enumerate(SIGMA):
        Np = NP[i_sig]
        fn = f"pi/pi_{q}_{n_shares}_{sigma}_{combif}_{n_attacks[i_sig]}.npy"
        with open(fn, "wb") as f:
            np.save(f, Np)
        res_holder = []
        for n_profiling in Np:
            res = compute_pi(q, sigma, n_coeffs, n_shares, n_profiling, n_attacks[i_sig], combif, n_splits=NSPLITS[i_sig], n_reps=100, n_proc=25)
            with open(fn, "ab") as f:
                np.save(f, np.array(res))
            res_holder.append(np.array(res).mean())
        plt.figure(figsize=(12, 12))
        plt.plot(Np, res_holder)
        plt.scatter(Np, res_holder)
        plt.savefig(f"pi/pi_{q}_{n_shares}_{sigma}_{combif}_{n_attacks[i_sig]}.png")
if __name__ == '__main__':
    SIGMA = [0.51672043, 1.63401346, 5.16720427]
    q = 3329
    n_shares = 3
    n_coeffs = 256
    sr_run()

    # n_profiling = 15000000
    # n_splits = 100
    # res = compute_pi(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits)
    # # res = compute_pi_seq(q, sigma, n_coeffs, n_shares, n_profiling, n_attack, combif, n_splits, n_reps=20, n_proc=5)
    # print(np.array(res).mean())
