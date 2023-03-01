import numpy as np
from scalib.attacks import SASCAGraph

import matplotlib.pyplot as plt
from gen_data import *
import pickle




def gen_graph(n_shares):
    graph_desc = '''NC 3329\nVAR SINGLE S\n'''
    prop = '''PROPERTY S ='''
    for i in range(n_shares):
        graph_desc += f"VAR MULTI X{i}\n" #add share
        prop +=  f" X{i}\n" if (i == n_shares - 1) else f" X{i} +" #add prop
    graph_desc += prop
    return graph_desc

def gen_leakage(shares_val, sigma):
    n_shares, batch_size, dim = shares_val.shape
    return {f"X{i}": HW(share_val) + np.random.normal(0, sigma, (batch_size, dim)) for i, share_val in enumerate(shares_val)}

def pdf_normal(x, mu, sigma):
    return np.exp(-(x-mu)**2/(2*sigma**2)) / (sigma * np.sqrt(2*np.pi))

def pdf_l_given_s(l, q, sigma=0.1):
    value_set = np.arange(q)
    hw_set = HW(value_set)
    pdf = pdf_normal(l, hw_set, sigma)


    return pdf/pdf.sum(axis=1, keepdims=True)


def ent(resX):
    """
    Computes the entropy
    """
    return np.nansum(-(np.log2(resX) * resX), axis=1).mean()

def exp_run(n_shares, sigma):
    n_profiling = 1000
    q = 3329
    secret = np.random.randint(-2, 3, (1, 1))
    secret_batch = np.repeat(secret, n_profiling, axis=0)
    shares_val = gen_shares(n_profiling, secret_batch, dim=1, n_order=n_shares, q=q)
    leakage = gen_leakage(shares_val, sigma)
    pdf_shares = {share: pdf_l_given_s(share_leakage, q=q, sigma=sigma) for share, share_leakage in leakage.items()}

    graph_desc = gen_graph(n_shares)
    graph = SASCAGraph(graph_desc, n_profiling)
    for i in range(n_shares):
        graph.set_init_distribution(f"X{i}", pdf_shares[f"X{i}"])
    prior_S = np.zeros((1, q))
    prior_S[:, 0] = 1/5
    prior_S[:, 1] = 1/5
    prior_S[:, 2] = 1/5
    prior_S[:, -2] = 1/5
    prior_S[:, -1] = 1/5
    graph.set_init_distribution(f"S", prior_S)
    graph.run_bp(2)
    resX = graph.get_distribution("S")
    entX = ent(resX)
    return np.log2(5) - entX


if __name__ == '__main__':
    n_shares = [2, 3, 4, 5, 6]
    sigmas = np.linspace(0.001, 10, 100)
    n_run = 20
    MIs = {}
    for ns in n_shares:
        stat = {}
        MI = []
        for sigma in sigmas:
            avg_mi = 0
            for nr in range(n_run):
                print(f"{ns} shares, noise: {sigma} run: {nr}")
                res = exp_run(ns, sigma)
                avg_mi += res if res > 0 else 0
            avg_mi = avg_mi/n_run
            print(f"avg_mi: {avg_mi}")
            MI.append(math.log(avg_mi))
        MIs[f"{ns}"] = MI
        plt.plot(sigmas, MI, label=f"{ns} shares")
    plt.legend()
    plt.xlabel("sigma")
    plt.ylabel("log(MI)")
    plt.show()
    with open("MI_stat.pkl", "wb") as f:
        pickle.dump(MIs, f)
# for ns, MI in stat.items():
    #     print(ns, MI)
    #     plt.plot(sigmas, MI,  label=f"{ns} shares")
    #     plt.xlabel('sigma')
    #     plt.ylabel('log(MI)')
    # plt.legend()
    # plt.show()


    # n_shares = 4
    # n_attack = 2
    # n_profiling = 1000
    # sigma = 1
    # q = 3329
    #
    #
    #
    #
    # graph_desc = gen_graph(n_shares)
    # graph = SASCAGraph(graph_desc, n_profiling)
    # for i in range(n_shares):
    #     graph.set_init_distribution(f"X{i}", pdf_shares[f"X{i}"])
    # prior_S = np.zeros((1, q))
    # prior_S[:, 0] = 1/5
    # prior_S[:, 1] = 1/5
    # prior_S[:, 2] = 1/5
    # prior_S[:, -2] = 1/5
    # prior_S[:, -1] = 1/5
    # graph.set_init_distribution(f"S", prior_S)
    # graph.run_bp(n_attack)
    # resX = graph.get_distribution("S")
    # print(resX.shape)
    # plt.plot(resX[0])
    #
    #
    #
    # entX0 = ent(pdf_shares[f"X0"])
    # entX1 = ent(pdf_shares[f"X1"])
    # entX = ent(resX)
    # entX_ = ent(prior_S)
    # entrop_max = np.log2(q)
    # print(graph_desc)
    # print(entX, entX_)
    # print(np.log2(5) - entX, entrop_max - entX0, entrop_max - entX1)
